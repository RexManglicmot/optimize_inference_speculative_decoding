# app/benchmarks.py

import time
import csv
from pathlib import Path
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

def run_specdec(prompts, drafts, verifier, run_id="specdec"):
    """
    For each prompt AND for each draft:
      - draft generates k tokens (timed)
      - verifier generates k tokens on same input (timed)
      - naive acceptance: all k tokens 'accepted'
    Writes one CSV row per (prompt, draft).
    """
    v_id, v_model, v_tok = verifier
    draft_k = int(cfg["decoding"].get("draft_k", 4))
    device = cfg["models"].get("device", "cuda")

    rows = []
    pbar = tqdm(prompts, desc=f"Running {run_id}")
    with torch.inference_mode():
        for ex in pbar:
            input_ids = ex["input_ids"].unsqueeze(0).to(device, non_blocking=True)
            attn = ex["attention_mask"].unsqueeze(0).to(device, non_blocking=True)

            for draft_id, d_model, d_tok in drafts:
                # Draft proposes k tokens
                _, draft_dt = _time_call(
                    d_model.generate,
                    input_ids=input_ids,
                    attention_mask=attn,
                    max_new_tokens=draft_k,
                    do_sample=False,
                    pad_token_id=d_tok.eos_token_id,
                )

                # Verifier "checks" k tokens (here: matched-length generate for timing)
                _, ver_dt = _time_call(
                    v_model.generate,
                    input_ids=input_ids,
                    attention_mask=attn,
                    max_new_tokens=draft_k,
                    do_sample=False,
                    pad_token_id=v_tok.eos_token_id,
                )

                tokens_total = draft_k
                tokens_accepted = draft_k  # naive acceptance (all pass)
                total_time = draft_dt + ver_dt
                tps = (tokens_total / total_time) if total_time > 0 else 0.0
                acc_rate = tokens_accepted / tokens_total if tokens_total else 0.0

                rows.append({
                    "id": ex["id"],
                    "run": run_id,
                    "draft_id": draft_id,
                    "verifier_id": v_id,
                    "tokens_total": tokens_total,
                    "tokens_accepted": tokens_accepted,
                    "latency_draft_s": draft_dt,
                    "latency_verifier_s": ver_dt,
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
    max_new = int(cfg["decoding"].get("max_new_tokens", 128))
    device = cfg["models"].get("device", "cuda")

    rows = []
    pbar = tqdm(prompts, desc="Running baseline")
    with torch.inference_mode():
        for ex in pbar:
            input_ids = ex["input_ids"].unsqueeze(0).to(device, non_blocking=True)
            attn = ex["attention_mask"].unsqueeze(0).to(device, non_blocking=True)

            _, ver_dt = _time_call(
                v_model.generate,
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=v_tok.eos_token_id,
            )

            tokens_total = max_new
            tps = (tokens_total / ver_dt) if ver_dt > 0 else 0.0

            rows.append({
                "id": ex["id"],
                "run": "baseline",
                "draft_id": "",                 # N/A in baseline
                "verifier_id": v_id,
                "tokens_total": tokens_total,
                "tokens_accepted": tokens_total, # baseline 'accepts' its own tokens
                "latency_draft_s": 0.0,          # N/A
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
