# app/viz.py

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from app.config import cfg

OUTDIR = Path(cfg["project"]["output_dir"])
SUMMARY = OUTDIR / "summary_table_rex_version.csv"

def _bar_plot(x_labels, y_values, title, y_label, outfile):
    plt.figure(figsize=(10, 5))
    plt.bar(x_labels, y_values)
    plt.xticks(rotation=30, ha="right")
    plt.title(title)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

def main():
    if not SUMMARY.exists():
        raise FileNotFoundError(f"Missing summary CSV: {SUMMARY}")
    df = pd.read_csv(SUMMARY)

    # Keep column names consistent
    cols = {
        "draft": "draft_id",
        "lat_p50": "latency_p50_sec_per_token",
        "lat_p95": "latency_p95_sec_per_token",
        "tps_p50": "throughput_tokens_per_sec_p50",
        # Rex added
        "tps_p95": "throughput_tokens_per_sec_p95",
        "acc_mean": "acceptance_rate_mean",
        "speedup": "speedup_x_vs_baseline",
        "disagree": "disagreement_rate"  # optional
    }
    for k, v in cols.items():
        if v not in df.columns:
            df[v] = float("nan")

    # Order bars (try speedup desc if present; else by latency p50 asc)
    if df[cols["speedup"]].notna().any():
        df = df.sort_values(by=cols["speedup"], ascending=False)
    else:
        df = df.sort_values(by=cols["lat_p50"], ascending=True)

    labels = df[cols["draft"]].astype(str).tolist()

    # 1) Latency p50 (sec/token)
    _bar_plot(
        labels,
        df[cols["lat_p50"]].tolist(),
        "Latency (p50) — seconds per token",
        "sec/token",
        OUTDIR / "latency_p50_bar_3.png",
    )

    # 2) Latency p95 (sec/token)
    _bar_plot(
        labels,
        df[cols["lat_p95"]].tolist(),
        "Latency (p95) — seconds per token",
        "sec/token",
        OUTDIR / "latency_p95_bar_3.png",
    )

    # 3) Throughput (tokens/sec, p50)
    _bar_plot(
        labels,
        df[cols["tps_p50"]].tolist(),
        "Throughput (p50)",
        "tokens/sec",
        OUTDIR / "throughput_bar_3_p50.png",
    )

    # Rex added
    # 3) Throughput (tokens/sec, p50)
    _bar_plot(
        labels,
        df[cols["tps_p95"]].tolist(),
        "Throughput (p95)",
        "tokens/sec",
        OUTDIR / "throughput_bar_3_p95.png",
    )
    
    # 4) Acceptance rate (%)
    acc_pct = (df[cols["acc_mean"]] * 100.0).tolist()
    _bar_plot(
        labels,
        acc_pct,
        "Verifier Acceptance Rate",
        "percent",
        OUTDIR / "acceptance_bar_3.png",
    )

    # 5) Speedup factor (× vs baseline)
    _bar_plot(
        labels,
        df[cols["speedup"]].tolist(),
        "Speedup (× vs baseline)",
        "×",
        OUTDIR / "speedup_bar_3.png",
    )

    # 6) Disagreement rate (%) — optional if present
    if df[cols["disagree"]].notna().any():
        _bar_plot(
            labels,
            (df[cols["disagree"]] * 100.0).tolist(),
            "Disagreement Rate (SpecDec vs Baseline)",
            "percent",
            OUTDIR / "disagreement_bar_3.png",
        )

    print("✅ Plots saved to:", OUTDIR.resolve())

if __name__ == "__main__":
    main()
