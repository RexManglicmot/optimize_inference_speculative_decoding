# app/eval.py
from pathlib import Path
import pandas as pd
import numpy as np

from app.config import cfg

OUTDIR = Path(cfg["project"]["output_dir"])
SPECDEC_CSV = OUTDIR / "results_specdec.csv"
BASELINE_CSV = OUTDIR / "results_baseline.csv"
SUMMARY_CSV = OUTDIR / "summary_table_rex_version_withcost.csv"

# ---- Simple pricing (one multiplier) ----
PRICING = cfg.get("pricing", {})
USD_PER_GPU_HOUR = float(PRICING.get("gpu_hour_usd", PRICING.get("usd_per_gpu_hour", 1.25)))
AVG_OUTPUT_TOKENS = float(PRICING.get("avg_output_tokens", 180))
USE_PERCENTILE = str(PRICING.get("use_percentile", "p50")).lower()

USD_PER_SEC = USD_PER_GPU_HOUR / 3600.0
K = USD_PER_SEC * AVG_OUTPUT_TOKENS  # <-- single multiplier

def _read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

def percentile(s: pd.Series, q: float) -> float:
    if len(s) == 0:
        return float("nan")
    return float(np.nanpercentile(s.to_numpy(dtype=float), q))

def main():
    spec = _read_csv_safe(SPECDEC_CSV)
    base = _read_csv_safe(BASELINE_CSV)

    if spec.empty:
        raise FileNotFoundError(f"No specdec results found at {SPECDEC_CSV}")
    if base.empty:
        print(f"⚠️  No baseline file at {BASELINE_CSV}; speedup will be NaN.")
        base = pd.DataFrame(columns=["id", "latency_verifier_s", "tokens_total"])

    # ---- Normalize per-row timings ----
    spec = spec.copy()
    spec["total_time_s"] = spec["latency_draft_s"].astype(float) + spec["latency_verifier_s"].astype(float)
    spec["tokens_total"] = spec["tokens_total"].astype(float).replace(0, np.nan)
    spec["sec_per_token"] = spec["total_time_s"] / spec["tokens_total"]
    spec["toks_per_sec"] = 1.0 / spec["sec_per_token"]

    base = base.copy()
    if not base.empty:
        base["latency_verifier_s"] = base["latency_verifier_s"].astype(float)
        base["tokens_total"] = base["tokens_total"].astype(float).replace(0, np.nan)
        base["sec_per_token"] = base["latency_verifier_s"] / base["tokens_total"]
        base["toks_per_sec"] = 1.0 / base["sec_per_token"]

        base_med_spt = percentile(base["sec_per_token"], 50)
        base_p95_spt = percentile(base["sec_per_token"], 95)
        base_med_tps = percentile(base["toks_per_sec"], 50)
        base_p95_tps = percentile(base["toks_per_sec"], 95)
    else:
        base_med_spt = np.nan
        base_p95_spt = np.nan
        base_med_tps = np.nan
        base_p95_tps = np.nan

    # ---- Aggregate per draft ----
    def agg_per_draft(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for draft_id, g in df.groupby("draft_id", dropna=False):
            lat_p50 = percentile(g["sec_per_token"], 50)
            lat_p95 = percentile(g["sec_per_token"], 95)
            tps_p50 = percentile(g["toks_per_sec"], 50)
            tps_p95 = percentile(g["toks_per_sec"], 95)
            #acc_mean = float(np.nanmean(g["acceptance_rate"].to_numpy(dtype=float))) if "acceptance_rate" in g else float("nan")
            speedup_x = (base_med_spt / lat_p50) if (not np.isnan(base_med_spt) and not np.isnan(lat_p50) and lat_p50 > 0) else np.nan
            rows.append({
                "draft_id": draft_id,
                "latency_p50_sec_per_token": lat_p50,
                "latency_p95_sec_per_token": lat_p95,
                "throughput_tokens_per_sec_p50": tps_p50,
                "throughput_tokens_per_sec_p95": tps_p95,
                # "acceptance_rate_mean": acc_mean,
                "speedup_x_vs_baseline": speedup_x,
            })
        return pd.DataFrame(rows)

    summary = agg_per_draft(spec)

    # Add baseline row
    baseline_row = {
        "draft_id": "BASELINE(verifier_only)",
        "latency_p50_sec_per_token": base_med_spt,
        "latency_p95_sec_per_token": base_p95_spt,
        "throughput_tokens_per_sec_p50": base_med_tps,
        "throughput_tokens_per_sec_p95": base_p95_tps,
        # "acceptance_rate_mean": 1.0 if not np.isnan(base_med_spt) else np.nan,
        "speedup_x_vs_baseline": 1.0 if not np.isnan(base_med_spt) else np.nan,
    }
    summary = pd.concat([summary, pd.DataFrame([baseline_row])], ignore_index=True)

    # ---- Cost columns (one-multiplier; uses chosen percentile) ----
    chosen_lat_col = "latency_p95_sec_per_token" if USE_PERCENTILE == "p95" else "latency_p50_sec_per_token"
    baseline_lat = base_p95_spt if USE_PERCENTILE == "p95" else base_med_spt

    summary["cost_per_query_usd"] = summary[chosen_lat_col].astype(float) * K
    summary["savings_per_query_usd"] = (baseline_lat - summary[chosen_lat_col].astype(float)) * K
    summary["usd_per_token_saved"] = (baseline_lat - summary[chosen_lat_col].astype(float)) * USD_PER_SEC

    OUTDIR.mkdir(parents=True, exist_ok=True)
    summary = summary.sort_values(by="speedup_x_vs_baseline", ascending=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"✅ Wrote summary: {SUMMARY_CSV}")
    print(summary)

if __name__ == "__main__":
    main()
