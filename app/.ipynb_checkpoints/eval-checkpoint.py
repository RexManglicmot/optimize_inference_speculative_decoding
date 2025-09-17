# app/eval.py

from pathlib import Path
import pandas as pd
import numpy as np

from app.config import cfg

OUTDIR = Path(cfg["project"]["output_dir"])
SPECDEC_CSV = OUTDIR / "results_specdec.csv"
BASELINE_CSV = OUTDIR / "results_baseline.csv"
SUMMARY_CSV = OUTDIR / "summary_table_rex_version.csv"

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
        print(f" No baseline file at {BASELINE_CSV}; speedup will be NaN.")
        base = pd.DataFrame(columns=["id", "latency_verifier_s", "tokens_total"])

    # ---- Normalize per-row timings ----
    # SpecDec: total_time = draft + verifier (for draft_k tokens)
    spec = spec.copy()
    spec["total_time_s"] = spec["latency_draft_s"].astype(float) + spec["latency_verifier_s"].astype(float)
    spec["tokens_total"] = spec["tokens_total"].astype(float).replace(0, np.nan)
    spec["sec_per_token"] = spec["total_time_s"] / spec["tokens_total"]
    spec["toks_per_sec"] = 1.0 / spec["sec_per_token"]

    # Baseline: verifier-only time for its tokens (usually max_new_tokens)
    base = base.copy()
    if not base.empty:
        base["latency_verifier_s"] = base["latency_verifier_s"].astype(float)
        base["tokens_total"] = base["tokens_total"].astype(float).replace(0, np.nan)
        base["sec_per_token"] = base["latency_verifier_s"] / base["tokens_total"]
        base["toks_per_sec"] = 1.0 / base["sec_per_token"]

        # Aggregate baseline per prompt (should already be 1 row/prompt)
        # Then reduce to scalar baselines for speedup comparisons
        base_med_spt = percentile(base["sec_per_token"], 50)
        base_p95_spt = percentile(base["sec_per_token"], 95)
        base_med_tps = percentile(base["toks_per_sec"], 50)
        base_p95_tps = percentile(base["toks_per_sec"], 95)
    else:
        base_med_spt = np.nan
        base_p95_spt = np.nan
        base_med_tps = np.nan

    # ---- Aggregate per draft ----
    def agg_per_draft(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for draft_id, g in df.groupby("draft_id", dropna=False):
            lat_p50 = percentile(g["sec_per_token"], 50)
            lat_p95 = percentile(g["sec_per_token"], 95)
            tps_p50 = percentile(g["toks_per_sec"], 50)
            tps_p95 = percentile(g["toks_per_sec"], 95)
            speedup_x = (base_med_spt / lat_p50) if (not np.isnan(base_med_spt) and not np.isnan(lat_p50) and lat_p50 > 0) else np.nan
            rows.append({
                "draft_id": draft_id,
                "latency_p50_sec_per_token": lat_p50,
                "latency_p95_sec_per_token": lat_p95,
                "throughput_tokens_per_sec_p50": tps_p50,
                "throughput_tokens_per_sec_p95": tps_p95,
                "speedup_x_vs_baseline": speedup_x,
            })
        return pd.DataFrame(rows)

    summary = agg_per_draft(spec)

    # Add baseline reference as a row (optional, helpful for tables/plots)
    summary = summary.sort_values(by="speedup_x_vs_baseline", ascending=False)
    baseline_row = {
        "draft_id": "BASELINE(verifier_only)",
        "latency_p50_sec_per_token": base_med_spt,
        "latency_p95_sec_per_token": base_p95_spt,
        "throughput_tokens_per_sec_p50": base_med_tps,
        "throughput_tokens_per_sec_p95": base_p95_tps,
        "speedup_x_vs_baseline": 1.0 if not np.isnan(base_med_spt) else np.nan,
    }
    summary = pd.concat([summary, pd.DataFrame([baseline_row])], ignore_index=True)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f" Wrote summary: {SUMMARY_CSV}")
    print(summary)

if __name__ == "__main__":
    main()
