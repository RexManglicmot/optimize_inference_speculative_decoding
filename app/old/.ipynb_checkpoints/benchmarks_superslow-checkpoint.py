# app/benchmarks.py

import time
import csv
from pathlib import Path
from typing import Tuple
import torch
from tqdm import tqdm

from app.config import cfg
from app.dataset import load_prompts
from app.models import load_models

OUTDIR = Path(cfg["project"]["output_dir"])
OUTDIR.mkdir(parents=True, exist_ok=True)

def _time_call(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return out, (t1 - t0)

def _longest_agreeing_prefix_len(
    v_model,
    v_tok,
    context_ids: torch.Tensor,   # [1, T]
    proposed_ids: torch.Tensor,  # [1, K]
    device: str
) -> Tuple[int, float]:
    """
    Run a single verifier forward over (context + proposed) and
    return how many proposed tokens match verifier's greedy next-token.
    """
    with torch.inference_mode():
        concat = torch.cat([context_ids, proposed_ids], dim=1)  # [1, T+K]
        attn = torch.ones_like(concat)
        # One forward to get logits for all positions
        (_, vt) = _time_call(
            v_model,
            input_ids=concat.to(device, non_blocking=True),
            attention_mask=attn.to(device, non_blocking=True),
        )
        logits = v_model(
            input_ids=concat.to(device, non_blocking=True),
            attention_mask=attn.to(device, non_blocking=True),
        ).logits  # [1, T+K, V]
        # Greedy predictions for each next token
        greedy = torch.argmax(logits[:, :-1, :], dim=-1)  # predicts token at t from position t-1
        T = context_ids.size(1)
        K = proposed_ids.size(1)
        agree = 0
        for i in range(K):
            # compare predicted next token at position T-1+i to proposed token at position T+i
            if greedy[0, T - 1 + i].item() == proposed_ids[0, i].item():
                agree += 1
            else:
                break
    return agree, vt

def run_specdec(prompts, drafts, verifier, run_id="specdec"):
    """
    True speculative decoding:
      - loop until max_new_tokens are PRODUCED
      - each hop: draft proposes k tokens
      - verifier checks in one forward; accept longest prefix
      - on first mismatch, verifier generates 1 token to correct
    One CSV row per (prompt, draft).
    """
    v_id, v_model, v_tok = verifier
    device = cfg["models"].get("device", "cuda")
    draft_k = int(cfg["decoding"].get("draft_k", 16))
    max_new = int(cfg["decoding"].get("max_new_tokens", 256))

    rows = []
    pbar = tqdm(prompts, desc=f"Running {run_id}")
    for ex in pbar:
        # Base context from dataset (already tokenized on CPU)
        base_ctx = ex["input_ids"].unsqueeze(0).to(device, non_blocking=True)

        for draft_id, d_model, d_tok in drafts:
            produced = 0
            proposed_total = 0
            accepted_total = 0
            draft_time = 0.0
            verifier_time = 0.0

            # Running context (on device)
            context_ids = base_ctx.clone()

            with torch.inference_mode():
                while produced < max_new:
                    # ---- DRAFT proposes k tokens (greedy; cheap/fast) ----
                    gen_args = dict(
                        input_ids=context_ids,
                        max_new_tokens=min(draft_k, max_new - produced),
                        do_sample=False,
                        pad_token_id=d_tok.eos_token_id,
                    )
                    _, dt = _time_call(d_model.generate, **gen_args)
                    draft_time += dt

                    draft_seq = d_model.generate(**gen_args)  # full sequence with new tokens appended
                    proposed = draft_seq[:, -min(draft_k, max_new - produced):]  # [1, k']
                    k = proposed.size(1)
                    proposed_total += k

                    # ---- VERIFIER checks in ONE forward; accept prefix ----
                    agree_len, vt = _longest_agreeing_prefix_len(
                        v_model, v_tok, context_ids, proposed, device
                    )
                    verifier_time += vt

                    # Accept agreed tokens
                    if agree_len > 0:
                        accepted = proposed[:, :agree_len]
                        context_ids = torch.cat([context_ids, accepted], dim=1)
                        accepted_total += agree_len
                        produced += agree_len

                    # On mismatch, have VERIFIER generate 1 corrective token
                    if agree_len < k and produced < max_new:
                        (_, vt2) = _time_call(
                            v_model.generate,
                            input_ids=context_ids,
                            max_new_tokens=1,
                            do_sample=False,
                            pad_token_id=v_tok.eos_token_id,
                        )
                        verifier_time += vt2
                        corr = v_model.generate(
                            input_ids=context_ids,
                            max_new_tokens=1,
                            do_sample=False,
                            pad_token_id=v_tok.eos_token_id,
                        )
                        next_tok = corr[:, -1:]
                        context_ids = torch.cat([context_ids, next_tok], dim=1)
                        produced += 1  # correction counts toward produced tokens

            total_time = draft_time + verifier_time
            tps = (produced / total_time) if total_time > 0 else 0.0
            acc_rate = (accepted_total / proposed_total) if proposed_total > 0 else 0.0

            rows.append({
                "id": ex["id"],
                "run": run_id,
                "draft_id": draft_id,
                "verifier_id": v_id,
                "tokens_total": produced,                # produced tokens (up to max_new)
                "tokens_accepted": accepted_total,       # accepted-from-draft count
                "latency_draft_s": draft_time,
                "latency_verifier_s": verifier_time,
                "throughput_tokens_per_s": tps,
                "acceptance_rate": acc_rate,
            })

    out_csv = OUTDIR / f"results_{run_id}.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id", "run", "draft_id", "verifier_id",
                "tokens_total", "tokens_accepted",
                "latency_draft_s", "latency_verifier_s",
                "throughput_tokens_per_s", "acceptance_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ Wrote {len(rows)} rows to {out_csv}")
    return rows

def run_baseline(prompts, verifier):
    """
    Verifier-only baseline: one row per prompt.
    """
    v_id, v_model, v_tok = verifier
    device = cfg["models"].get("device", "cuda")
    max_new = int(cfg["decoding"].get("max_new_tokens", 256))

    rows = []
    pbar = tqdm(prompts, desc="Running baseline")
    with torch.inference_mode():
        for ex in pbar:
            input_ids = ex["input_ids"].unsqueeze(0).to(device, non_blocking=True)

            (_, ver_dt) = _time_call(
                v_model.generate,
                input_ids=input_ids,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=v_tok.eos_token_id,
            )
            out = v_model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=v_tok.eos_token_id,
            )
            produced = out.size(1) - input_ids.size(1)

            tps = (produced / ver_dt) if ver_dt > 0 else 0.0

            rows.append({
                "id": ex["id"],
                "run": "baseline",
                "draft_id": "",
                "verifier_id": v_id,
                "tokens_total": produced,
                "tokens_accepted": produced,
                "latency_draft_s": 0.0,
                "latency_verifier_s": ver_dt,
                "throughput_tokens_per_s": tps,
                "acceptance_rate": 1.0,
            })

    out_csv = OUTDIR / "results_baseline.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id", "run", "draft_id", "verifier_id",
                "tokens_total", "tokens_accepted",
                "latency_draft_s", "latency_verifier_s",
                "throughput_tokens_per_s", "acceptance_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ Wrote {len(rows)} rows to {out_csv}")
    return rows

if __name__ == "__main__":
    print("🔹 Loading dataset...")
    prompts = load_prompts()
    print(f"Loaded {len(prompts)} prompts")

    print("🔹 Loading models...")
    drafts, verifier = load_models()

    if cfg["decoding"].get("speculative_enabled", True):
        spec_rows = run_specdec(prompts, drafts, verifier, run_id="specdec")
        print(f"Finished specdec: {len(spec_rows)} rows "
              f"(~{len(prompts)} prompts × {len(drafts)} drafts)")

    if cfg["evaluation"]["baseline"].get("run_verifier_only", False):
        base_rows = run_baseline(prompts, verifier)
        print(f"Finished baseline: {len(base_rows)} rows (~{len(prompts)} prompts)")
