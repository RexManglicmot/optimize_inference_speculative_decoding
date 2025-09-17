# app/benchmarks.py
#
# Speculative decoding (robust mode for GPT-J):
#  - Always rebuild verifier past_key_values from the full current context each hop.
#  - Check k proposed tokens in one forward; accept prefix; if mismatch, do 1 corrective step.
#  - Pass attention_mask everywhere.
# Outputs:
#  - ./outputs/results_specdec.csv  (one row per prompt × draft)
#  - ./outputs/results_baseline.csv (one row per prompt) if enabled or not skipped.

import time
import csv
import argparse
from pathlib import Path
from typing import Tuple

import torch
from tqdm import tqdm

from app.config import cfg
from app.dataset import load_prompts
from app.models import load_models

OUTDIR = Path(cfg["project"]["output_dir"])
OUTDIR.mkdir(parents=True, exist_ok=True)

def _now() -> float:
    return time.perf_counter()

@torch.inference_mode()
def _encode_context_get_past(v_model, context_ids: torch.Tensor) -> Tuple[tuple, float]:
    """Encode full context to obtain fresh past_key_values. Returns (past, elapsed_s)."""
    attn = torch.ones_like(context_ids)
    t0 = _now()
    out = v_model(input_ids=context_ids, attention_mask=attn, use_cache=True)
    dt = _now() - t0
    return out.past_key_values, dt

@torch.inference_mode()
def _verifier_check_k(v_model, proposed: torch.Tensor, past, ctx_len: int) -> Tuple[torch.Tensor, float]:
    """
    Check K proposed tokens using existing past. Returns (logits [1,K,V], elapsed_s).
    Attention mask must be length (ctx_len + K).
    """
    K = proposed.size(1)
    attn_mask = torch.ones((1, ctx_len + K), dtype=proposed.dtype, device=proposed.device)
    t0 = _now()
    out = v_model(
        input_ids=proposed,
        attention_mask=attn_mask,
        past_key_values=past,
        use_cache=True,
    )
    dt = _now() - t0
    return out.logits, dt

def _agreeing_prefix(logits: torch.Tensor, proposed: torch.Tensor) -> int:
    greedy = torch.argmax(logits, dim=-1)  # [1, K]
    K = proposed.size(1)
    agree = 0
    for i in range(K):
        if greedy[0, i].item() == proposed[0, i].item():
            agree += 1
        else:
            break
    return agree

def run_specdec(prompts, drafts, verifier, run_id="specdec"):
    """
    Robust speculative decoding that avoids KV/mask desyncs by rebuilding past each hop.
    One CSV row per (prompt, draft).
    """
    v_id, v_model, v_tok = verifier
    device = cfg["models"].get("device", "cuda")
    draft_k = int(cfg["decoding"].get("draft_k", 32))
    max_new = int(cfg["decoding"].get("max_new_tokens", 256))

    rows = []
    pbar = tqdm(prompts, desc=f"Running {run_id}")

    for ex in pbar:
        base_ctx = ex["input_ids"].unsqueeze(0).to(device, non_blocking=True)

        for draft_id, d_model, d_tok in drafts:
            produced = 0
            proposed_total = 0
            accepted_total = 0
            draft_time = 0.0
            verifier_time = 0.0

            context_ids = base_ctx.clone()  # running context passed to the draft

            while produced < max_new:
                # --- DRAFT proposes k' tokens (timed) ---
                k = min(draft_k, max_new - produced)
                gen_args = dict(
                    input_ids=context_ids,
                    attention_mask=torch.ones_like(context_ids),
                    max_new_tokens=k,
                    do_sample=False,
                    pad_token_id=d_tok.eos_token_id,
                )
                t0 = _now()
                draft_seq = d_model.generate(**gen_args)
                draft_dt = _now() - t0
                draft_time += draft_dt

                proposed = draft_seq[:, -k:]  # [1, k]
                proposed_total += k

                # --- VERIFIER: rebuild past on full current context (robust) ---
                past, dt_ctx = _encode_context_get_past(v_model, context_ids)
                verifier_time += dt_ctx
                ctx_len = context_ids.size(1)

                # --- Check the proposed tokens using past ---
                logits, dt_chk = _verifier_check_k(v_model, proposed, past, ctx_len)
                verifier_time += dt_chk
                agree_len = _agreeing_prefix(logits, proposed)

                # Accept the agreeing prefix
                if agree_len > 0:
                    accepted = proposed[:, :agree_len]
                    context_ids = torch.cat([context_ids, accepted], dim=1)
                    accepted_total += agree_len
                    produced += agree_len

                # If mismatch and still need tokens, get ONE corrective token (greedy)
                if agree_len < k and produced < max_new:
                    # Rebuild past for the updated context before correction
                    past, dt_ctx2 = _encode_context_get_past(v_model, context_ids)
                    verifier_time += dt_ctx2

                    # One-step greedy correction: forward a single dummy step
                    # Construct attention mask for (ctx_len + 1) inside generate by passing full context
                    t0 = _now()
                    corr = v_model.generate(
                        input_ids=context_ids,
                        max_new_tokens=1,
                        do_sample=False,
                        pad_token_id=v_tok.eos_token_id,
                    )
                    ver_dt = _now() - t0
                    verifier_time += ver_dt

                    next_tok = corr[:, -1:]  # [1,1]
                    context_ids = torch.cat([context_ids, next_tok], dim=1)
                    produced += 1

            total_time = draft_time + verifier_time
            tps = (produced / total_time) if total_time > 0 else 0.0
            acc_rate = (accepted_total / proposed_total) if proposed_total > 0 else 0.0

            rows.append({
                "id": ex["id"],
                "run": run_id,
                "draft_id": draft_id,
                "verifier_id": v_id,
                "tokens_total": produced,
                "tokens_accepted": accepted_total,
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

@torch.inference_mode()
def run_baseline(prompts, verifier):
    """
    Verifier-only baseline (greedy decode). One row per prompt.
    """
    v_id, v_model, v_tok = verifier
    device = cfg["models"].get("device", "cuda")
    max_new = int(cfg["decoding"].get("max_new_tokens", 256))

    rows = []
    pbar = tqdm(prompts, desc="Running baseline")
    for ex in pbar:
        input_ids = ex["input_ids"].unsqueeze(0).to(device, non_blocking=True)

        t0 = _now()
        out = v_model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=v_tok.eos_token_id,
        )
        ver_dt = _now() - t0

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

def parse_args():
    ap = argparse.ArgumentParser(description="Speculative decoding benchmarks (robust)")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of prompts (overrides config sample_size at runtime)")
    ap.add_argument("--no-baseline", action="store_true", help="Skip verifier-only baseline run")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()

    print("🔹 Loading dataset...")
    prompts = load_prompts()
    if args.limit is not None:
        prompts = prompts[: args.limit]
    print(f"Loaded {len(prompts)} prompts")

    print("🔹 Loading models...")
    drafts, verifier = load_models()

    if cfg["decoding"].get("speculative_enabled", True):
        spec_rows = run_specdec(prompts, drafts, verifier, run_id="specdec")
        print(f"Finished specdec: {len(spec_rows)} rows (~{len(prompts)} prompts × {len(drafts)} drafts)")

    run_base = cfg["evaluation"]["baseline"].get("run_verifier_only", False) and not args.no_baseline
    if run_base:
        base_rows = run_baseline(prompts, verifier)
        print(f"Finished baseline: {len(base_rows)} rows (~{len(prompts)} prompts)")
