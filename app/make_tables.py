# app/make_tables.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import re

# --- Fixed locations ---
OUTDIR = Path("outputs")
SUMMARY_CSV = OUTDIR / "summary_table_rex_version.csv"
AUDIT_CSV   = OUTDIR / "acceptance_audit.csv"
OUT_PERF_MD = OUTDIR / "table_performance.md"
OUT_AUD_MD  = OUTDIR / "table_audit.md"

# --------------- helpers ---------------
def df_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        vals = [("" if pd.isna(v) else str(v)) for v in row.tolist()]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"

def fmt_num(v: float, prec: int = 4) -> str:
    if pd.isna(v): return ""
    return f"{float(v):.{prec}f}"

def fmt_pct_if_01(series: pd.Series, prec: int = 1) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().between(0.0, 1.5).all():
        return s.map(lambda x: f"{x*100.0:.{prec}f}%")
    return s.map(lambda x: fmt_num(x, 4))

# --------------- table builders ---------------
def performance_table(summary: pd.DataFrame) -> pd.DataFrame:
    # pick label col
    label_col = next((c for c in ["draft_id","model","name","verifier_or_model","id"]
                      if c in summary.columns), None)
    if label_col is None:
        raise KeyError("No label column in summary CSV (expected draft_id/model/name/verifier_or_model/id).")

    s = summary.copy()
    if "speedup_x_vs_baseline" in s.columns:
        s = s.sort_values("speedup_x_vs_baseline", ascending=False)
    elif "latency_p50_sec_per_token" in s.columns:
        s = s.sort_values("latency_p50_sec_per_token", ascending=True)

    cols_want = [
        label_col,
        "latency_p50_sec_per_token",
        "latency_p95_sec_per_token",
        "throughput_tokens_per_sec_p50",
        "throughput_tokens_per_sec_p95",
        "speedup_x_vs_baseline",
    ]
    cols = [c for c in cols_want if c in s.columns]
    t = s[cols].copy()

    # numeric format
    for c in ["latency_p50_sec_per_token","latency_p95_sec_per_token"]:
        if c in t.columns:
            t[c] = pd.to_numeric(t[c], errors="coerce").map(lambda x: fmt_num(x, 4))
    for c in ["throughput_tokens_per_sec_p50","throughput_tokens_per_sec_p95"]:
        if c in t.columns:
            t[c] = pd.to_numeric(t[c], errors="coerce").map(lambda x: fmt_num(x, 1))
    if "speedup_x_vs_baseline" in t.columns:
        t["speedup_x_vs_baseline"] = pd.to_numeric(t["speedup_x_vs_baseline"], errors="coerce").map(lambda x: fmt_num(x, 2))

    return t.rename(columns={
        label_col: "model",
        "latency_p50_sec_per_token": "latency_p50 (s/token)",
        "latency_p95_sec_per_token": "latency_p95 (s/token)",
        "throughput_tokens_per_sec_p50": "throughput_p50 (tok/s)",
        "throughput_tokens_per_sec_p95": "throughput_p95 (tok/s)",
        "speedup_x_vs_baseline": "speedup (×)",
    })

def audit_table(audit: pd.DataFrame) -> pd.DataFrame:
    cols = ["draft_id", "samples"]
    # acceptance mean if present
    for name in ["acceptance_rate_mean", "acceptance_mean", "acceptance_rate"]:
        if name in audit.columns:
            cols.append(name); break
    # p50/p95 (any spelling)
    cols += [c for c in audit.columns if re.fullmatch(r"acceptance(_rate)?_p50", c)]
    cols += [c for c in audit.columns if re.fullmatch(r"acceptance(_rate)?_p95", c)]
    # optional extras
    if "disagreement_rate" in audit.columns:
        cols.append("disagreement_rate")
    sem_col = "semantic_sim_mean" if "semantic_sim_mean" in audit.columns else (
        "semantic_similarity_mean" if "semantic_similarity_mean" in audit.columns else None
    )
    if sem_col: cols.append(sem_col)

    cols = [c for c in cols if c in audit.columns]
    t = audit[cols].copy()

    # percentage-style for 0..1 columns (except draft_id/samples)
    for c in t.columns:
        if c in ("draft_id", "samples"):
            continue
        t[c] = fmt_pct_if_01(t[c])

    if sem_col:
        t = t.rename(columns={sem_col: "semantic_similarity_mean"})
    return t.rename(columns={
        "disagreement_rate": "disagreement",
        "acceptance_rate_mean": "acceptance_mean",
    })

# --------------- main ---------------
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing summary CSV: {SUMMARY_CSV}")
    if not AUDIT_CSV.exists():
        raise FileNotFoundError(f"Missing audit CSV: {AUDIT_CSV}")

    summary = pd.read_csv(SUMMARY_CSV)
    audit   = pd.read_csv(AUDIT_CSV)

    perf_md  = df_to_markdown(performance_table(summary))
    audit_md = df_to_markdown(audit_table(audit))

    OUT_PERF_MD.write_text(perf_md, encoding="utf-8")
    OUT_AUD_MD.write_text(audit_md, encoding="utf-8")

    print("✅ Wrote:", OUT_PERF_MD)
    print("✅ Wrote:", OUT_AUD_MD)

if __name__ == "__main__":
    main()
