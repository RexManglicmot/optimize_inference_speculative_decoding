# app/viz.py
from pathlib import Path
import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from app.config import cfg

OUTDIR = Path(cfg["project"]["output_dir"])
DEFAULT_SUMMARY = OUTDIR / "summary_table_rex_version.csv"
DEFAULT_AUDIT   = OUTDIR / "acceptance_audit.csv"



# --- Styling knobs
BAR_WIDTH_SINGLE   = 0.55   # slimmer single-series bars
HEADROOM_SINGLE    = 0.20   # +20% above tallest bar for labels
TOTAL_GROUP_WIDTH  = 0.72   # fraction of tick width covered by all grouped bars
HEADROOM_GROUP     = 0.20   # +20% headroom for grouped plots
LABEL_OFFSET_FRAC  = 0.025  # ~2.5% of y-span above each bar


# ---------- utils 
def _ensure_outdir():
    OUTDIR.mkdir(parents=True, exist_ok=True)

def _fmt(v: float) -> str:
    if pd.isna(v):
        return "NaN"
    a = abs(v)
    if a >= 100:   return f"{v:.0f}"
    if a >= 10:    return f"{v:.1f}"
    if a >= 1:     return f"{v:.2f}"
    if a >= 0.1:   return f"{v:.3f}"
    return f"{v:.4f}"

def _attach_value_labels(ax, rects, values, y_offset_frac=LABEL_OFFSET_FRAC):
    """
    Place value labels on top of bars with a small vertical offset.
    IMPORTANT: call this AFTER setting ylim so offsets use the final y-span.
    """
    ymin, ymax = ax.get_ylim()
    yrange = (ymax - ymin) or 1.0
    for r, val in zip(rects, values):
        if pd.isna(val):
            continue
        ax.text(
            r.get_x() + r.get_width() / 2.0,
            r.get_height() + y_offset_frac * yrange,
            _fmt(val),
            ha="center",
            va="bottom",
            fontsize=10,
        )

def _bar_plot(x_labels, y_values, title, y_label, outfile):
    plt.figure(figsize=(10, 5))
    ax = plt.gca()

    # Draw slimmer bars
    rects = ax.bar(x_labels, y_values, width=BAR_WIDTH_SINGLE)

    # Titles/axes
    ax.set_title(title)
    ax.set_ylabel(y_label)
    plt.xticks(rotation=30, ha="right")

    # Guarantee headroom BEFORE placing labels
    finite_vals = [float(v) for v in y_values if isinstance(v, (int, float)) and not pd.isna(v)]
    if finite_vals:
        top = max(finite_vals)
        ax.set_ylim(0, top * (1.0 + HEADROOM_SINGLE))

    # Now add labels (offset uses final ylim)
    _attach_value_labels(ax, rects, y_values)

    plt.tight_layout()
    plt.savefig(outfile, dpi=140, bbox_inches="tight")
    plt.close()


def _group_bar_plot(x_labels, series_dict, title, y_label, outfile):
    """
    Grouped bars with slimmer total width and headroom for labels.
    series_dict: {legend_label: list_of_values}
    """
    n_groups = len(x_labels)
    n_series = max(1, len(series_dict))

    # Geometry: keep groups slim by covering only a portion of tick width
    total_width = TOTAL_GROUP_WIDTH
    bar_width = total_width / n_series
    x = np.arange(n_groups)

    plt.figure(figsize=(12, 5))
    ax = plt.gca()

    rects_all = []
    keys = list(series_dict.keys())
    for i, k in enumerate(keys):
        vals = series_dict[k]
        offs = x - total_width/2 + i*bar_width + bar_width/2
        rects = ax.bar(offs, vals, width=bar_width, label=k)
        rects_all.append((rects, vals))

    # Titles/axes
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xticks(x, x_labels, rotation=30, ha="right")
    ax.legend()

    # Headroom BEFORE adding labels (use max across all series)
    all_vals = []
    for _, vals in rects_all:
        all_vals.extend([float(v) for v in vals if isinstance(v, (int, float)) and not pd.isna(v)])
    if all_vals:
        top = max(all_vals)
        ax.set_ylim(0, top * (1.0 + HEADROOM_GROUP))

    # Labels after ylim
    for rects, vals in rects_all:
        _attach_value_labels(ax, rects, vals)

    plt.tight_layout()
    plt.savefig(outfile, dpi=140, bbox_inches="tight")
    plt.close()


    
# ---------- loaders & normalization ----------
def _resolve_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _normalize_key_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    """
    Return a unified key series (string) from any of the candidate columns.
    Trims spaces and lowercases for robust joining.
    """
    col = _resolve_col(df, candidates)
    if col is None:
        raise KeyError(f"Could not find join key among columns: {candidates}")
    return df[col].astype(str).str.strip()


def _load_summary(path: str | None) -> pd.DataFrame:
    p = Path(path) if path else DEFAULT_SUMMARY
    if not p.exists():
        raise FileNotFoundError(f"Summary CSV not found: {p}")
    return pd.read_csv(p)

def _load_audit(path: str | None) -> pd.DataFrame:
    p = Path(path) if path else DEFAULT_AUDIT
    if not p.exists():
        raise FileNotFoundError(f"Audit CSV not found: {p}")
    return pd.read_csv(p)

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=str, default=None, help=f"Path to summary CSV (default: {DEFAULT_SUMMARY})")
    parser.add_argument("--audit", type=str, default=None, help=f"Path to acceptance/audit CSV (default: {DEFAULT_AUDIT})")
    args = parser.parse_args()

    _ensure_outdir()

    # Load both CSVs
    df_sum = _load_summary(args.summary)
    df_aud = _load_audit(args.audit)

    # Define column maps
    sum_cols = {
        "key":    ["draft_id", "model", "name", "verifier_or_model", "id"],
        "lat50":  ["latency_p50_sec_per_token"],
        "lat95":  ["latency_p95_sec_per_token"],
        "tps50":  ["throughput_tokens_per_sec_p50"],
        "tps95":  ["throughput_tokens_per_sec_p95"],
        "speed":  ["speedup_x_vs_baseline"],
    }
    aud_cols = {
        "key":    ["draft_id", "model", "name", "verifier_or_model", "id"],
        "acc":    ["acceptance_rate_mean"],
        "dis":    ["disagreement_rate", "diagreement_rate"],  # typo-safe
        "sem":    ["semantic_sim_mean", "semantic_similarity_mean"],
    }

    # Normalize join keys
    df_sum = df_sum.copy()
    df_aud = df_aud.copy()
    df_sum["_join_key"] = _normalize_key_series(df_sum, sum_cols["key"]).str.lower()
    df_aud["_join_key"] = _normalize_key_series(df_aud, aud_cols["key"]).str.lower()

    # Select & rename needed columns for clarity before merge
    def pick(df, colmap):
        out = {"_join_key": df["_join_key"]}
        for k, cands in colmap.items():
            if k == "key":
                continue
            col = _resolve_col(df, cands)
            if col is None:
                out[k] = pd.Series([np.nan]*len(df))
            else:
                out[k] = df[col]
        return pd.DataFrame(out)

    left = pd.DataFrame({
        "_join_key": df_sum["_join_key"],
        "label": _normalize_key_series(df_sum, sum_cols["key"])  # preserve nice label
    })
    left = left.join(pick(df_sum, sum_cols).set_index("_join_key"), on="_join_key")

    right = pick(df_aud, aud_cols)

    # Merge summary (left) + audit (right)
    df = left.merge(right, how="left", on="_join_key", suffixes=("", "_aud"))

    # Sort: by speedup desc if available else by latency p50 asc
    if df["speed"].notna().any():
        df = df.sort_values(by="speed", ascending=False)
    else:
        df = df.sort_values(by="lat50", ascending=True)

    labels = df["label"].astype(str).tolist()

    # ---------- plots ----------
    # 1) Grouped Latency p50/p95
    if df[["lat50","lat95"]].notna().any().any():
        _group_bar_plot(
            labels,
            {"p50": df["lat50"].tolist(), "p95": df["lat95"].tolist()},
            "Latency — seconds per token (p50 vs p95)",
            "sec/token",
            OUTDIR / "latency_grouped_p50_p95.png",
        )

    # 2) Grouped Throughput p50/p95
    if df[["tps50","tps95"]].notna().any().any():
        _group_bar_plot(
            labels,
            {"p50": df["tps50"].tolist(), "p95": df["tps95"].tolist()},
            "Throughput — tokens/sec (p50 vs p95)",
            "tokens/sec",
            OUTDIR / "throughput_grouped_p50_p95.png",
        )

    # 3) Speedup (× vs baseline)
    if df["speed"].notna().any():
        _bar_plot(
            labels,
            df["speed"].tolist(),
            "Speedup (× vs baseline)",
            "×",
            OUTDIR / "speedup_bar.png",
        )
    
    # 4) Acceptance rate (%)
    if df["acc"].notna().any():
        _bar_plot(
            labels,
            (df["acc"] * 100.0).tolist(),
            "Verifier Acceptance Rate",
            "percent",
            OUTDIR / "acceptance_bar.png",
        )


    print("Plots saved to:", OUTDIR.resolve())

if __name__ == "__main__":
    main()
