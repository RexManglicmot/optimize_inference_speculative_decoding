# app/viz_2.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from app.config import cfg

OUTDIR = Path(cfg["project"]["output_dir"])
SUMMARY = OUTDIR / "summary_table_2.csv"

def _bar(x, y, title, ylabel, outfile):
    plt.figure(figsize=(10, 5))
    plt.bar(x, y)
    plt.xticks(rotation=30, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

def main():
    if not SUMMARY.exists():
        raise FileNotFoundError(f"Missing summary CSV: {SUMMARY}")
    df = pd.read_csv(SUMMARY)

    # Put baseline last and sort by speedup desc for readability
    is_base = df["draft_id"].astype(str).str.contains("BASELINE", case=False, na=False)
    nonbase = df[~is_base].copy()
    base = df[is_base].copy()

    if nonbase["speedup_x_vs_baseline"].notna().any():
        nonbase = nonbase.sort_values("speedup_x_vs_baseline", ascending=False)
    labels = nonbase["draft_id"].tolist() + base["draft_id"].tolist()
    df_plot = pd.concat([nonbase, base], ignore_index=True)

    # Latency p50 / p95
    _bar(
        labels,
        df_plot["latency_p50_sec_per_token"].tolist(),
        "Latency p50 (sec/token)",
        "sec/token",
        OUTDIR / "latency_p50_p95_2.png",  # will be overwritten by p95 next
    )
    # Overlay a second figure for p95 (separate file)
    _bar(
        labels,
        df_plot["latency_p95_sec_per_token"].tolist(),
        "Latency p95 (sec/token)",
        "sec/token",
        OUTDIR / "latency_p95_2.png",
    )

    # Throughput p50 / p95
    _bar(
        labels,
        df_plot["throughput_tokens_per_sec_p50"].tolist(),
        "Throughput p50 (tokens/sec)",
        "tokens/sec",
        OUTDIR / "throughput_2.png",
    )
    if "throughput_tokens_per_sec_p95" in df_plot.columns:
        _bar(
            labels,
            df_plot["throughput_tokens_per_sec_p95"].tolist(),
            "Throughput p95 (tokens/sec)",
            "tokens/sec",
            OUTDIR / "throughput_p95_2.png",
        )

    # Acceptance rate (%)
    _bar(
        labels,
        (df_plot["acceptance_rate_mean"] * 100.0).tolist(),
        "Verifier Acceptance Rate (%)",
        "percent",
        OUTDIR / "acceptance_rate_2.png",
    )

    # Disagreement rate (%)
    if "disagreement_rate" in df_plot.columns:
        _bar(
            labels,
            (df_plot["disagreement_rate"] * 100.0).tolist(),
            "Disagreement Rate (%)",
            "percent",
            OUTDIR / "disagreement_rate_2.png",
        )

    # Semantic similarity (0–1)
    if "semantic_sim_mean" in df_plot.columns:
        _bar(
            labels,
            df_plot["semantic_sim_mean"].tolist(),
            "Semantic Similarity (SpecDec vs Baseline)",
            "score (0–1)",
            OUTDIR / "semantic_similarity_2.png",
        )

    # Speedup (×)
    _bar(
        labels,
        df_plot["speedup_x_vs_baseline"].tolist(),
        "Speedup (× vs baseline)",
        "×",
        OUTDIR / "speedup_vs_baseline_2.png",
    )

    # Cost per query (relative)
    _bar(
        labels,
        df_plot["cost_per_query_rel"].tolist(),
        "Relative Cost per Query (lower is better)",
        "relative cost",
        OUTDIR / "cost_per_query_rel_2.png",
    )

    print("✅ Plots saved to:", OUTDIR.resolve())

if __name__ == "__main__":
    main()




# """
# viz_2.py
# Read outputs/summary_results_2.csv and write plots with _2 suffix.

# Creates:
# - latency_p50_p95_2.png
# - throughput_2.png
# - acceptance_rate_2.png
# - disagreement_rate_2.png
# - semantic_similarity_2.png
# - speedup_vs_baseline_2.png
# """

# import os
# import pandas as pd
# import matplotlib.pyplot as plt

# OUTDIR = "./outputs"

# def _savefig(name):
#     path = os.path.join(OUTDIR, name)
#     plt.savefig(path, dpi=150, bbox_inches="tight")
#     plt.close()
#     return path

# def _order(df):
#     # Order with baseline first, then drafts alpha by size-ish if present
#     base = df[df["variant"]=="baseline"]
#     spec = df[df["variant"]=="specdec"].copy()
#     # Try to sort drafts by model size hinted in name; fallback alpha
#     spec = spec.sort_values(by="draft_id")
#     return pd.concat([base, spec], ignore_index=True)

# def plot_latency(df):
#     dfp = _order(df)
#     x = range(len(dfp))
#     fig, ax = plt.subplots()
#     ax.bar(x, dfp["latency_p50_ms"], label="p50")
#     ax.bar(x, dfp["latency_p95_ms"], label="p95", alpha=0.6)
#     ax.set_xticks(x)
#     ax.set_xticklabels(dfp["draft_id"], rotation=30, ha="right")
#     ax.set_ylabel("Latency (ms per request)")
#     ax.set_title("Latency (p50/p95)")
#     ax.legend()
#     _savefig("latency_p50_p95_2.png")

# def plot_throughput(df):
#     dfp = _order(df)
#     x = range(len(dfp))
#     fig, ax = plt.subplots()
#     ax.bar(x, dfp["throughput_tok_s"])
#     ax.set_xticks(x)
#     ax.set_xticklabels(dfp["draft_id"], rotation=30, ha="right")
#     ax.set_ylabel("Tokens / second")
#     ax.set_title("Throughput")
#     _savefig("throughput_2.png")

# def plot_acceptance(df):
#     dfp = df[df["variant"]=="specdec"].copy()
#     if "acceptance_rate_mean" not in dfp.columns:
#         return
#     x = range(len(dfp))
#     fig, ax = plt.subplots()
#     ax.bar(x, dfp["acceptance_rate_mean"])
#     ax.set_xticks(x)
#     ax.set_xticklabels(dfp["draft_id"], rotation=30, ha="right")
#     ax.set_ylabel("Acceptance rate (0–1)")
#     ax.set_title("Verifier Acceptance Rate (mean)")
#     _savefig("acceptance_rate_2.png")

# def plot_disagreement(df):
#     dfp = df[df["variant"]=="specdec"].copy()
#     if "disagreement_rate" not in dfp.columns:
#         return
#     x = range(len(dfp))
#     fig, ax = plt.subplots()
#     ax.bar(x, dfp["disagreement_rate"])
#     ax.set_xticks(x)
#     ax.set_xticklabels(dfp["draft_id"], rotation=30, ha="right")
#     ax.set_ylabel("Disagreement rate (0–1)")
#     ax.set_title("SpecDec vs Verifier-only Disagreement")
#     _savefig("disagreement_rate_2.png")

# def plot_semantic(df):
#     dfp = df[df["variant"]=="specdec"].copy()
#     if "semantic_sim_mean" not in dfp.columns:
#         return
#     x = range(len(dfp))
#     fig, ax = plt.subplots()
#     ax.bar(x, dfp["semantic_sim_mean"])
#     ax.set_xticks(x)
#     ax.set_xticklabels(dfp["draft_id"], rotation=30, ha="right")
#     ax.set_ylabel("Cosine similarity (0–1)")
#     ax.set_title("Semantic Similarity (SpecDec vs Baseline)")
#     _savefig("semantic_similarity_2.png")

# def plot_speedup(df):
#     # Speedup vs baseline p50
#     base = df[(df["variant"]=="baseline")].copy()
#     if base.empty:
#         return
#     base_p50 = float(base["latency_p50_ms"].iloc[0])

#     dfp = df.copy()
#     dfp["speedup_x"] = base_p50 / dfp["latency_p50_ms"]

#     x = range(len(dfp))
#     fig, ax = plt.subplots()
#     ax.bar(x, dfp["speedup_x"])
#     ax.set_xticks(x)
#     ax.set_xticklabels(dfp["draft_id"], rotation=30, ha="right")
#     ax.set_ylabel("× Speedup vs baseline (p50 latency)")
#     ax.set_title("Speedup Factor")
#     _savefig("speedup_vs_baseline_2.png")

# def main():
#     path = os.path.join(OUTDIR, "summary_results_2.csv")
#     if not os.path.exists(path):
#         raise FileNotFoundError("summary_results_2.csv not found — run eval_2.py first.")
#     df = pd.read_csv(path)

#     # Fill NAs to avoid matplotlib issues
#     for c in ["latency_p50_ms","latency_p95_ms","throughput_tok_s","acceptance_rate_mean","disagreement_rate","semantic_sim_mean"]:
#         if c in df.columns:
#             df[c] = pd.to_numeric(df[c], errors="coerce")

#     plot_latency(df)
#     plot_throughput(df)
#     plot_acceptance(df)
#     plot_disagreement(df)
#     plot_semantic(df)
#     plot_speedup(df)

#     print("✅ Plots saved to:", os.path.abspath(OUTDIR))

# if __name__ == "__main__":
#     main()
