# app/eval_2.py
from pathlib import Path
import math
import pandas as pd

from app.config import cfg

OUTDIR = Path(cfg["project"]["output_dir"])
SPEC = OUTDIR / "results_specdec.csv"
BASE = OUTDIR / "results_baseline.csv"
AUDT = OUTDIR / "acceptance_audit.csv"
OUT  = OUTDIR / "summary_table_2.csv"

# --- helpers ---------------------------------------------------------------

def _read_csv_safe(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f"Missing CSV: {p}")
    return pd.read_csv(p)

def _rename_if_present(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    present = {k: v for k, v in mapping.items() if k in df.columns}
    if present:
        df = df.rename(columns=present)
    return df

# --- main ------------------------------------------------------------------

def main():
    spec = _read_csv_safe(SPEC)
    base = _read_csv_safe(BASE)
    aud  = _read_csv_safe(AUDT)

    # Normalize column names we care about
    # (accept both your earlier and current headers)
    spec = _rename_if_present(
        spec,
        {
            "draft": "draft_id",
            "lat_p50": "latency_p50_sec_per_token",
            "lat_p95": "latency_p95_sec_per_token",
            "tps_p50": "throughput_tokens_per_sec_p50",
            "tps_p95": "throughput_tokens_per_sec_p95",
            "acc_mean": "acceptance_rate_mean",
            "speedup": "speedup_x_vs_baseline",
            "disagree": "disagreement_rate",
            "sem_sim": "semantic_sim_mean",
        },
    )

    base = _rename_if_present(
        base,
        {
            "draft": "draft_id",
            "lat_p50": "latency_p50_sec_per_token",
            "lat_p95": "latency_p95_sec_per_token",
            "tps_p50": "throughput_tokens_per_sec_p50",
            "tps_p95": "throughput_tokens_per_sec_p95",
        },
    )

    aud = _rename_if_present(
        aud,
        {
            "draft": "draft_id",
            "acceptance_rate": "acceptance_rate_mean",
            "semantic_similarity": "semantic_sim_mean",
        },
    )

    # Keep only per-draft rows (drop the baseline row if it’s present)
    # Baseline file can contain a “baseline/BASELINE(verifier_only)” row.
    base_row = None
    # try to locate baseline in the baseline CSV
    for cand in ["baseline", "BASELINE(verifier_only)", "verifier_only", "BASELINE"]:
        if "draft_id" in base.columns and cand in set(base["draft_id"].astype(str)):
            base_row = (
                base[base["draft_id"].astype(str) == cand]
                .iloc[0]
                .to_dict()
            )
            break
    # Fallback: if baseline.csv only has one row, treat it as baseline
    if base_row is None and len(base) == 1:
        base_row = base.iloc[0].to_dict()

    # Join acceptance / disagreement / semantic stats from the audit
    # The audit CSV already holds one row per draft_id.
    keep_aud_cols = [
        "draft_id",
        "acceptance_rate_mean",
        "disagreement_rate",
        "semantic_sim_mean",
    ]
    aud = aud[keep_aud_cols].drop_duplicates("draft_id")

    # Merge onto spec rows
    merged = spec.merge(aud, on="draft_id", how="left", suffixes=("", "_aud"))

    # Ensure all requested columns exist, even if NaN
    for col in [
        "latency_p50_sec_per_token",
        "latency_p95_sec_per_token",
        "throughput_tokens_per_sec_p50",
        "throughput_tokens_per_sec_p95",
        "acceptance_rate_mean",
        "speedup_x_vs_baseline",
        "disagreement_rate",
        "semantic_sim_mean",
    ]:
        if col not in merged.columns:
            merged[col] = math.nan

    # If throughput_p95 missing, leave NaN (we don’t guess here)
    if "throughput_tokens_per_sec_p95" not in merged.columns:
        merged["throughput_tokens_per_sec_p95"] = math.nan

    # Derive relative cost: 1 / speedup (baseline cost = 1.0)
    def inv_or_nan(x):
        try:
            return 1.0 / float(x) if float(x) > 0 else math.nan
        except Exception:
            return math.nan

    merged["cost_per_query_rel"] = merged["speedup_x_vs_baseline"].apply(inv_or_nan)

    # Order columns exactly as requested
    ordered = merged[
        [
            "draft_id",
            "latency_p50_sec_per_token",
            "latency_p95_sec_per_token",
            "throughput_tokens_per_sec_p50",
            "throughput_tokens_per_sec_p95",
            "acceptance_rate_mean",
            "speedup_x_vs_baseline",
            "disagreement_rate",
            "semantic_sim_mean",
            "cost_per_query_rel",
        ]
    ].copy()

    # Round for readability
    num_cols = [
        c
        for c in ordered.columns
        if c != "draft_id"
    ]
    ordered[num_cols] = ordered[num_cols].astype(float).round(6)

    # Append a final baseline row for reference (if we found it)
    if base_row is not None:
        baseline_out = {
            "draft_id": "BASELINE(verifier_only)",
            "latency_p50_sec_per_token": base_row.get("latency_p50_sec_per_token", math.nan),
            "latency_p95_sec_per_token": base_row.get("latency_p95_sec_per_token", math.nan),
            "throughput_tokens_per_sec_p50": base_row.get("throughput_tokens_per_sec_p50", math.nan),
            "throughput_tokens_per_sec_p95": base_row.get("throughput_tokens_per_sec_p95", math.nan),
            "acceptance_rate_mean": 1.0,  # trivially 1.0 for verifier-only
            "speedup_x_vs_baseline": 1.0,
            "disagreement_rate": 0.0,
            "semantic_sim_mean": 1.0,
            "cost_per_query_rel": 1.0,
        }
        ordered = pd.concat([ordered, pd.DataFrame([baseline_out])], ignore_index=True)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    ordered.to_csv(OUT, index=False)
    print(f"✅ Wrote {OUT} with shape {ordered.shape}")

if __name__ == "__main__":
    main()
