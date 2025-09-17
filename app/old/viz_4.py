# app/viz.py

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from app.config import cfg

OUTDIR = Path(cfg["project"]["output_dir"])
SUMMARY = OUTDIR / "summary_table_rex_version.csv"


def _ensure_outdir():
    OUTDIR.mkdir(parents=True, exist_ok=True)


def _fmt(v: float) -> str:
    """Pretty format numbers for bar labels."""
    if pd.isna(v):
        return "NaN"
    # Use fewer decimals for big numbers, more for small ones
    av = abs(v)
    if av >= 100:
        return f"{v:.0f}"
    elif av >= 10:
        return f"{v:.1f}"
    elif av >= 1:
        return f"{v:.2f}"
    elif av >= 0.1:
        return f"{v:.3f}"
    else:
        return f"{v:.4f}"


# def _attach_value_labels(ax, rects, values, y_offset_factor=0.01):
#     """Place value labels on top of bars with a small vertical offset."""
#     ylim = ax.get_ylim()
#     yrange = ylim[1] - ylim[0]
#     for rect, val in zip(rects, values):
#         if pd.isna(val):
#             continue
#         height = rect.get_height()
#         ax.text(
#             rect.get_x() + rect.get_width() / 2.0,
#             height + y_offset_factor * yrange,
#             _fmt(val),
#             ha="center",
#             va="bottom",
#             fontsize=9,
#             rotation=0,
#         )

def _attach_value_labels(ax, rects, values, y_offset_factor=0.015):
    ymin, ymax = ax.get_ylim()
    yrange = (ymax - ymin) or 1.0
    for r, val in zip(rects, values):
        if pd.isna(val):
            continue
        ax.text(
            r.get_x() + r.get_width()/2.0,
            r.get_height() + y_offset_factor*yrange,
            _fmt(val),
            ha="center", va="bottom", fontsize=9
        )



# def _bar_plot(x_labels, y_values, title, y_label, outfile):
#     plt.figure(figsize=(10, 5))
#     ax = plt.gca()
#     rects = ax.bar(x_labels, y_values)
#     ax.set_title(title)
#     ax.set_ylabel(y_label)
#     plt.xticks(rotation=30, ha="right")
#     plt.tight_layout()
#     _attach_value_labels(ax, rects, y_values)
#     plt.savefig(outfile, dpi=140, bbox_inches="tight")
#     plt.close()

def _bar_plot(x_labels, y_values, title, y_label, outfile):
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    rects = ax.bar(x_labels, y_values, width=0.6)   # slimmer bars
    ax.set_title(title)
    ax.set_ylabel(y_label)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    # Add labels
    _attach_value_labels(ax, rects, y_values)

    # Expand ylim to give label headroom
    ymin, ymax = ax.get_ylim()
    if ymax > 0:
        ax.set_ylim(ymin, ymax * 1.10)  # 10% taller headroom

    plt.savefig(outfile, dpi=140, bbox_inches="tight")
    plt.close()


def _group_bar_plot(x_labels, series_dict, title, y_label, outfile):
    """
    Grouped bar chart.
    series_dict: Ordered mapping {legend_label: list_of_values aligned with x_labels}
    """
    n_groups = len(x_labels)
    n_series = len(series_dict)
    assert n_series >= 1, "series_dict must have at least one series"

    # bar geometry
    total_width = 0.8
    bar_width = total_width / n_series
    x = np.arange(n_groups)

    plt.figure(figsize=(12, 5))
    ax = plt.gca()

    rects_all = []
    legends = list(series_dict.keys())
    for i, name in enumerate(legends):
        offsets = x - total_width / 2 + i * bar_width + bar_width / 2
        vals = series_dict[name]
        rects = ax.bar(offsets, vals, width=bar_width, label=name)
        rects_all.append((rects, vals))

    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xticks(x, x_labels, rotation=30, ha="right")
    ax.legend()
    plt.tight_layout()

    # Add labels on top of each bar
    for rects, vals in rects_all:
        _attach_value_labels(ax, rects, vals)

    plt.savefig(outfile, dpi=140, bbox_inches="tight")
    plt.close()


def main():
    _ensure_outdir()
    if not SUMMARY.exists():
        raise FileNotFoundError(f"Missing summary CSV: {SUMMARY}")
    df = pd.read_csv(SUMMARY)

    # Keep column names consistent
    cols = {
        "draft": "draft_id",
        "lat_p50": "latency_p50_sec_per_token",
        "lat_p95": "latency_p95_sec_per_token",
        "tps_p50": "throughput_tokens_per_sec_p50",
        "tps_p95": "throughput_tokens_per_sec_p95",
        "acc_mean": "acceptance_rate_mean",
        "speedup": "speedup_x_vs_baseline",
        "disagree": "disagreement_rate",  # optional
    }
    for _, v in cols.items():
        if v not in df.columns:
            df[v] = float("nan")

    # Order bars: speedup desc if present; else by latency p50 asc
    if df[cols["speedup"]].notna().any():
        df = df.sort_values(by=cols["speedup"], ascending=False)
    else:
        df = df.sort_values(by=cols["lat_p50"], ascending=True)

    labels = df[cols["draft"]].astype(str).tolist()

    # 1) Grouped Latency p50/p95 (sec/token)
    lat_p50_vals = df[cols["lat_p50"]].tolist()
    lat_p95_vals = df[cols["lat_p95"]].tolist()
    _group_bar_plot(
        labels,
        {
            "p50": lat_p50_vals,
            "p95": lat_p95_vals,
        },
        "Latency — seconds per token (p50 vs p95)",
        "sec/token",
        OUTDIR / "latency_grouped_p50_p95_4.png",
    )

    # 2) Grouped Throughput p50/p95 (tokens/sec)
    tps_p50_vals = df[cols["tps_p50"]].tolist()
    tps_p95_vals = df[cols["tps_p95"]].tolist()
    _group_bar_plot(
        labels,
        {
            "p50": tps_p50_vals,
            "p95": tps_p95_vals,
        },
        "Throughput — tokens/sec (p50 vs p95)",
        "tokens/sec",
        OUTDIR / "throughput_grouped_p50_p95_4.png",
    )

    # 3) Speedup (× vs baseline)
    _bar_plot(
        labels,
        df[cols["speedup"]].tolist(),
        "Speedup (× vs baseline)",
        "×",
        OUTDIR / "speedup_bar_4.png",
    )

    # # 4) Acceptance rate (%)
    # acc_pct = (df[cols["acc_mean"]] * 100.0).tolist()
    # _bar_plot(
    #     labels,
    #     acc_pct,
    #     "Verifier Acceptance Rate",
    #     "percent",
    #     OUTDIR / "acceptance_bar.png",
    #)

    # # 5) Disagreement rate (%) — optional if present
    # if df[cols["disagree"]].notna().any():
    #     _bar_plot(
    #         labels,
    #         (df[cols["disagree"]] * 100.0).tolist(),
    #         "Disagreement Rate (SpecDec vs Baseline)",
    #         "percent",
    #         OUTDIR / "disagreement_bar.png",
    #     )

    print("✅ Plots saved to:", OUTDIR.resolve())


if __name__ == "__main__":
    main()
