# app/benchmarks.py
# Fused speculative decoding (assistant_model) — requires batch_size == 1.
# Writes:
#   outputs/results_specdec.csv
#   outputs/results_baseline.csv (if enabled in config)

import csv
import time
from pathlib import Path
import argparse

import torch
from tqdm import tqdm

from app.config import cfg
from app.dataset import load_prompts
from app.models import load_models

# ---- Fast-but-safe math ----
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

OUTDIR = Path(cfg["project"]["output_dir"]).resolve()
OUTDIR.mkdir(parents=True, exist_ok=True)

DEVICE = cfg["models"].get("device", "cuda")
DEC = cfg.get("decoding", {})
MAX_NEW = int(DEC.get("max_new_tokens", 256))
DRAFT_K = int(DEC.get("draft_k", 64))
CTX_MAX = int(DEC.get("context_max_tokens", 512))
RUN_BASELINE = cfg["evaluation"]["baseline"].get("run_verifier_only", False)

def _sync_start():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()

def _sync_end(t0: float) -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - t0

def _write_csv(rows, path: Path):
    cols = [
        "id","run","draft_id","verifier_id",
        "tokens_total","tokens_accepted",
        "latency_draft_s","latency_verifier_s",
        "throughput_tokens_per_s","acceptance_rate"
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

@torch.inference_mode()
def run_specdec(prompts, drafts, verifier):
    """
    Speculative decoding using HF fused kernel:
      verifier.generate(assistant_model=draft, num_assistant_tokens=DRAFT_K)
    NOTE: HF requires batch_size == 1 for assisted generate.
    """
    v_id, v_model, v_tok = verifier
    rows = []

    pbar = tqdm(prompts, desc="SpecDec (fused, bs=1)")
    for ex in pbar:
        # Prepare single-sample context
        ctx = ex["input_ids"][:CTX_MAX].unsqueeze(0).to(DEVICE, non_blocking=True)
        if ctx.dtype != torch.long:
            ctx = ctx.long()
        attn = torch.ones_like(ctx)  # explicit mask (pad/eos warning silenced)

        for draft_id, d_model, _ in drafts:
            t0 = _sync_start()
            out = v_model.generate(
                input_ids=ctx,
                attention_mask=attn,
                max_new_tokens=MAX_NEW,
                do_sample=False,
                pad_token_id=v_tok.eos_token_id,
                use_cache=True,
                assistant_model=d_model,          # <<< fused speculative decoding
                num_assistant_tokens=DRAFT_K,
            )
            dt = _sync_end(t0)

            produced = int(out.size(1) - ctx.size(1))
            tps = (produced / dt) if dt > 0 else 0.0

            rows.append({
                "id": ex["id"],
                "run": "specdec",
                "draft_id": draft_id,
                "verifier_id": v_id,
                "tokens_total": produced,
                "tokens_accepted": produced,      # placeholder (kernel abstracts accept/reject)
                "latency_draft_s": 0.0,           # combined in verifier time
                "latency_verifier_s": dt,
                "throughput_tokens_per_s": tps,
                "acceptance_rate": 1.0,           # placeholder
            })

    path = OUTDIR / "results_specdec.csv"
    _write_csv(rows, path)
    print(f"✅ Wrote {len(rows)} rows to {path}")
    return rows

@torch.inference_mode()
def run_baseline(prompts, verifier):
    """Verifier-only baseline (bs=1 for apples-to-apples timing)."""
    v_id, v_model, v_tok = verifier
    rows = []

    pbar = tqdm(prompts, desc="Baseline (bs=1)")
    for ex in pbar:
        ctx = ex["input_ids"][:CTX_MAX].unsqueeze(0).to(DEVICE, non_blocking=True)
        if ctx.dtype != torch.long:
            ctx = ctx.long()
        attn = torch.ones_like(ctx)

        t0 = _sync_start()
        out = v_model.generate(
            input_ids=ctx,
            attention_mask=attn,
            max_new_tokens=MAX_NEW,
            do_sample=False,
            pad_token_id=v_tok.eos_token_id,
            use_cache=True,
        )
        dt = _sync_end(t0)

        produced = int(out.size(1) - ctx.size(1))
        tps = (produced / dt) if dt > 0 else 0.0

        rows.append({
            "id": ex["id"],
            "run": "baseline",
            "draft_id": "",
            "verifier_id": v_id,
            "tokens_total": produced,
            "tokens_accepted": produced,
            "latency_draft_s": 0.0,
            "latency_verifier_s": dt,
            "throughput_tokens_per_s": tps,
            "acceptance_rate": 1.0,
        })

    path = OUTDIR / "results_baseline.csv"
    _write_csv(rows, path)
    print(f"✅ Wrote {len(rows)} rows to {path}")
    return rows

def parse_args():
    ap = argparse.ArgumentParser(description="Fused SpecDec + Baseline (bs=1)")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of prompts")
    ap.add_argument("--no-baseline", action="store_true", help="Skip baseline")
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
        run_specdec(prompts, drafts, verifier)

    run_base = cfg["evaluation"]["baseline"].get("run_verifier_only", False) and not args.no_baseline
    if run_base:
        run_baseline(prompts, verifier)
