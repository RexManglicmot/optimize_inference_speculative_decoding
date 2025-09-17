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

def _measure_latency(func, *args, **kwargs):
    start = time.perf_counter()
    out = func(*args, **kwargs)
    end = time.perf_counter()
    return out, end - start

def run_specdec(prompts, drafts, verifier, run_id="specdec"):
    v_id, v_model, v_tok = verifier
    results = []

    draft_k = cfg["decoding"].get("draft_k", 4)
    max_new = cfg["decoding"].get("max_new_tokens", 128)
    device = cfg["models"].get("device", "cuda")

    for ex in tqdm(prompts, desc=f"Running {run_id}"):
        input_ids = ex["input_ids"].unsqueeze(0).to(device, non_blocking=True)
        attn = ex["attention_mask"].unsqueeze(0).to(device, non_blocking=True)

        total_tokens = 0
        accepted_tokens = 0
        latencies = []

        with torch.inference_mode():
            for draft_id, d_model, d_tok in drafts:
                # Draft proposes k tokens
                _, dt = _measure_latency(
                    d_model.generate,
                    input_ids=input_ids,
                    attention_mask=attn,
                    max_new_tokens=draft_k,
                    do_sample=False,
                    pad_token_id=d_tok.eos_token_id,
                )
                latencies.append(dt)
                total_tokens += draft_k

                # Verifier checks acceptance
                _, vt = _measure_latency(
                    v_model.generate,
                    input_ids=input_ids,
                    attention_mask=attn,
                    max_new_tokens=draft_k,
                    do_sample=False,
                    pad_token_id=v_tok.eos_token_id,
                )
                latencies.append(vt)

                # Naive acceptance heuristic:
                # here we just assume all draft tokens "pass"
                accepted_tokens += draft_k

        tps = total_tokens / sum(latencies) if latencies else 0.0
        acc_rate = accepted_tokens / total_tokens if total_tokens > 0 else 0.0

        results.append({
            "id": ex["id"],
            "run": run_id,
            "latencies": latencies,
            "tokens_total": total_tokens,
            "tokens_accepted": accepted_tokens,
            "throughput_tokens_per_s": tps,
            "acceptance_rate": acc_rate,
        })

    # Write CSV
    out_csv = OUTDIR / f"results_{run_id}.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id", "run", "tokens_total", "tokens_accepted",
                "throughput_tokens_per_s", "acceptance_rate", "latencies"
            ]
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"✅ Wrote {len(results)} rows to {out_csv}")
    return results

def run_baseline(prompts, verifier):
    v_id, v_model, v_tok = verifier
    results = []

    max_new = cfg["decoding"].get("max_new_tokens", 128)
    device = cfg["models"].get("device", "cuda")

    for ex in tqdm(prompts, desc="Running baseline"):
        input_ids = ex["input_ids"].unsqueeze(0).to(device, non_blocking=True)
        attn = ex["attention_mask"].unsqueeze(0).to(device, non_blocking=True)

        total_tokens = 0
        latencies = []

        with torch.inference_mode():
            _, vt = _measure_latency(
                v_model.generate,
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=v_tok.eos_token_id,
            )
            latencies.append(vt)
            total_tokens += max_new

        tps = total_tokens / sum(latencies) if latencies else 0.0

        results.append({
            "id": ex["id"],
            "run": "baseline",
            "tokens_total": total_tokens,
            "tokens_accepted": total_tokens,  # baseline accepts all its own tokens
            "throughput_tokens_per_s": tps,
            "acceptance_rate": 1.0,
            "latencies": latencies,
        })

    out_csv = OUTDIR / "results_baseline.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id", "run", "tokens_total", "tokens_accepted",
                "throughput_tokens_per_s", "acceptance_rate", "latencies"
            ]
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"✅ Wrote {len(results)} rows to {out_csv}")
    return results

if __name__ == "__main__":
    print("🔹 Loading dataset...")
    prompts = load_prompts()
    print(f"Loaded {len(prompts)} prompts")

    print("🔹 Loading models...")
    drafts, verifier = load_models()

    if cfg["decoding"].get("speculative_enabled", True):
        run_specdec(prompts, drafts, verifier, run_id="specdec")

    if cfg["evaluation"]["baseline"].get("run_verifier_only", False):
        run_baseline(prompts, verifier)
