# app/viz.py
from pathlib import Path
import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from app.config import cfg

OUTDIR = Path(cfg["project"]["output_dir"])
DEFAULT_SUMMARY = OUTDIR / "summary_table_rex_version_withcost.csv"

# ---------- utils ----------
def _ensure_outdir():
    OUTDIR.mkdir(parents=True, exist_ok=True)

def _fmt(v: float) -> str:
    if pd.isna(v): return "NaN"
    a = abs(v)
    if a >= 100: return f"{v:.0f}"
    if a >= 10:  return f"{v:.1f}"
    if a >= 1:   return f"{v:.2f}"
    if a >= 0.1: return f"{v:.3f}"
    return f"{v:.4f}"

def _attach_value_labels(ax, rects, values, y_offset_factor=0.02):
    ymin, ymax = ax.get_ylim()
    yrange = (ymax - ymin) or 1.0
    for r, val in zip(rects, values):
        if pd.isna(val): continue
        ax.text(
            r.get_x() + r.get_width()/2.0,
            r.get_height() + y_offset_factor*yrange,
            _fmt(val),
            ha="center", va="bottom", fontsize=9
        )

def _bar_plot(x_labels, y_values, title, y_label, outfile, *, bar_width=0.55, headroom=0.15):
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    rects = ax.bar(x_labels, y_values, width=bar_width)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    plt.xticks(rotation=30, ha="right")

    # headroom before placing labels
    finite_vals = [v for v in y_values if isinstance(v, (int, float)) and not pd.isna(v)]
    if finite_vals:
        top = max(finite_vals)
        ax.set_ylim(0, top * (1.0 + headroom))

    _attach_value_labels(ax, rects, y_values, y_offset_factor=0.02)
    plt.tight_layout()
    plt.savefig(outfile, dpi=140, bbox_inches="tight")
    plt.close()

def _group_bar_plot(x_labels, series_dict, title, y_label, outfile):
    n_groups = len(x_labels)
    n_series = len(series_dict)
    assert n_series >= 1, "series_dict must have at least one series"
    total_width = 0.8
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

    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xticks(x, x_labels, rotation=30, ha="right")
    ax.legend()
    plt.tight_layout()

    for rects, vals in rects_all:
        _attach_value_labels(ax, rects, vals)

    plt.savefig(outfile, dpi=140, bbox_inches="tight")
    plt.close()

# ---------- loaders & normalization ----------
def _resolve_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _normalize_key_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    col = _resolve_col(df, candidates)
    if col is None:
        raise KeyError(f"Could not find join key among columns: {candidates}")
    return df[col].astype(str).str.strip()

def _load_summary(path: str | None) -> pd.DataFrame:
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Summary CSV not found: {p}")
        return pd.read_csv(p)
    if DEFAULT_SUMMARY.exists():
        return pd.read_csv(DEFAULT_SUMMARY)
    raise FileNotFoundError(f"Missing summary CSV. Looked for {DEFAULT_SUMMARY}. Pass --summary <path>.")

def _find_default_audit() -> str | None:
    cands = sorted(glob.glob(str(OUTDIR / "acceptance_audit*.csv")))
    return cands[0] if cands else None

def _load_audit(path: str | None) -> pd.DataFrame:
    if not path:
        path = _find_default_audit()
        if not path:
            raise FileNotFoundError(
                "Missing acceptance/audit CSV. Place a file like 'acceptance_audit*.csv' in OUTDIR "
                "or pass --audit <path>."
            )
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Audit CSV not found: {p}")
    return pd.read_csv(p)

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=str, default=None, help="Path to summary CSV (latency/throughput/speedup)")
    parser.add_argument("--audit",   type=str, default=None, help="Path to acceptance/audit CSV")
    args = parser.parse_args()

    _ensure_outdir()

    df_sum = _load_summary(args.summary)
    df_aud = _load_audit(args.audit)

    # Column maps (now include cost columns from eval.py)
    sum_cols = {
        "key":    ["draft_id", "model", "name", "verifier_or_model", "id"],
        "lat50":  ["latency_p50_sec_per_token"],
        "lat95":  ["latency_p95_sec_per_token"],
        "tps50":  ["throughput_tokens_per_sec_p50"],
        "tps95":  ["throughput_tokens_per_sec_p95"],
        "speed":  ["speedup_x_vs_baseline"],
        "cost":   ["cost_per_query_usd"],
        "save":   ["savings_per_query_usd"],
        "usdspt": ["usd_per_token_saved"],
    }
    aud_cols = {
        "key":    ["draft_id", "model", "name", "verifier_or_model", "id"],
        "acc":    ["acceptance_rate_mean"],
        "dis":    ["disagreement_rate", "diagreement_rate"],
        "sem":    ["semantic_sim_mean", "semantic_similarity_mean"],
    }

    df_sum = df_sum.copy(); df_aud = df_aud.copy()
    df_sum["_join_key"] = _normalize_key_series(df_sum, sum_cols["key"]).str.lower()
    df_aud["_join_key"] = _normalize_key_series(df_aud, aud_cols["key"]).str.lower()

    def pick(df, colmap):
        out = {"_join_key": df["_join_key"]}
        for k, cands in colmap.items():
            if k == "key": continue
            col = _resolve_col(df, cands)
            out[k] = df[col] if col is not None else pd.Series([np.nan]*len(df))
        return pd.DataFrame(out)

    left = pd.DataFrame({
        "_join_key": df_sum["_join_key"],
        "label": _normalize_key_series(df_sum, sum_cols["key"])
    })
    left = left.join(pick(df_sum, sum_cols).set_index("_join_key"), on="_join_key")

    right = pick(df_aud, aud_cols)

    df = left.merge(right, how="left", on="_join_key", suffixes=("", "_aud"))

    if df["speed"].notna().any():
        df = df.sort_values(by="speed", ascending=False)
    else:
        df = df.sort_values(by="lat50", ascending=True)

    labels = df["label"].astype(str).tolist()

    # ---------- plots ----------
    if df[["lat50","lat95"]].notna().any().any():
        _group_bar_plot(
            labels,
            {"p50": df["lat50"].tolist(), "p95": df["lat95"].tolist()},
            "Latency — seconds per token (p50 vs p95)",
            "sec/token",
            OUTDIR / "latency_grouped_p50_p95_6.png",
        )

    if df[["tps50","tps95"]].notna().any().any():
        _group_bar_plot(
            labels,
            {"p50": df["tps50"].tolist(), "p95": df["tps95"].tolist()},
            "Throughput — tokens/sec (p50 vs p95)",
            "tokens/sec",
            OUTDIR / "throughput_grouped_p50_p95_6.png",
        )

    if df["speed"].notna().any():
        _bar_plot(
            labels,
            df["speed"].tolist(),
            "Speedup (× vs baseline)",
            "×",
            OUTDIR / "speedup_bar_6.png",
        )

    if df["acc"].notna().any():
        _bar_plot(
            labels,
            (df["acc"] * 100.0).tolist(),
            "Verifier Acceptance Rate",
            "percent",
            OUTDIR / "acceptance_bar_6.png",
        )

    if df["dis"].notna().any():
        _bar_plot(
            labels,
            (df["dis"] * 100.0).tolist(),
            "Disagreement Rate (Draft vs Verifier)",
            "percent",
            OUTDIR / "disagreement_bar_6.png",
        )

    if df["sem"].notna().any():
        vals = pd.to_numeric(df["sem"], errors="coerce")
        if vals.dropna().between(0.0, 1.5).all():
            plot_vals, ylab = (vals * 100.0).tolist(), "percent"
        else:
            plot_vals, ylab = vals.tolist(), "value"
        _bar_plot(
            labels,
            plot_vals,
            "Semantic Similarity (Mean)",
            ylab,
            OUTDIR / "semantic_similarity_bar_6.png",
        )

    # --- NEW: Cost plots from eval.py ---
    if "cost" in df and df["cost"].notna().any():
        _bar_plot(
            labels,
            pd.to_numeric(df["cost"], errors="coerce").tolist(),
            "Cost per Query",
            "USD/query",
            OUTDIR / "cost_per_query_bar_6.png",
        )

    if "save" in df and df["save"].notna().any():
        _bar_plot(
            labels,
            pd.to_numeric(df["save"], errors="coerce").tolist(),
            "Savings per Query vs Baseline",
            "USD/query",
            OUTDIR / "savings_vs_baseline_bar_6.png",
        )

    # Optional: $ per token saved
    if "usdspt" in df and df["usdspt"].notna().any():
        _bar_plot(
            labels,
            pd.to_numeric(df["usdspt"], errors="coerce").tolist(),
            "USD per Token Saved",
            "USD/token",
            OUTDIR / "usd_per_token_saved_bar_6.png",
        )

    print("✅ Plots saved to:", OUTDIR.resolve())

if __name__ == "__main__":
    main()
