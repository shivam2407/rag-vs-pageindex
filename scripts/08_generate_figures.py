"""
Script 08: Generate all publication figures.

Fig 1 — Grouped bar with 95% CI error bars: Judge score by domain (3 systems)
Fig 2 — Grouped bar: Token cost per question by domain
Fig 3 — Scatter: Accuracy gain (PI−RAG) vs token cost ratio per domain
Fig 4 — Document length analysis: accuracy gap across length terciles (RQ2)

All saved as PNG (200 dpi, for drafts) + PDF (vector, for IEEE LaTeX submission).
IEEE style: no unnecessary chart junk, color-blind-safe palette.
"""

import csv, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "results"
FIGS = ROOT / "results" / "figures"
FIGS.mkdir(exist_ok=True)

# Color-blind safe (Okabe-Ito palette)
C_BM25 = "#E69F00"   # amber
C_RAG  = "#0072B2"   # blue
C_PI   = "#D55E00"   # vermillion
SHORT  = {
    "Finance (FinanceBench)": "Finance",
    "Legal (CUAD)":           "Legal",
    "Science (QASPER)":       "Science",
    "Technology (TechQA)":    "Tech",
    "OVERALL":                "Overall",
}


def load_csv(path):
    if not Path(path).exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))

def _save(fig, name):
    fig.savefig(FIGS / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.pdf",           bbox_inches="tight")
    print(f"  Saved: {name}.png / .pdf")

def safe_float(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ── Figure 1: Accuracy by domain, 3 systems, with 95% CI ────────────────────

def fig1_accuracy_bars(rows):
    import matplotlib.pyplot as plt
    import numpy as np

    data   = [r for r in rows if r["domain"] != "OVERALL"
              and safe_float(r.get("rag_judge")) is not None]
    labels = [SHORT.get(r["domain"], r["domain"]) for r in data]
    x = np.arange(len(labels))
    w = 0.25

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (method, color, label) in enumerate(
        [("bm25","#E69F00","BM25"), ("rag","#0072B2","Dense RAG"), ("pageindex","#D55E00","PageIndex")]
    ):
        means  = [safe_float(r.get(f"{method}_judge"), 0) for r in data]
        lo_errs= [means[j] - safe_float(r.get(f"{method}_judge_lo"), means[j])
                  for j, r in enumerate(data)]
        hi_errs= [safe_float(r.get(f"{method}_judge_hi"), means[j]) - means[j]
                  for j, r in enumerate(data)]
        bars = ax.bar(x + (i-1)*w, means, w,
                      label=label, color=color, alpha=0.85,
                      yerr=[lo_errs, hi_errs], capsize=4,
                      error_kw={"elinewidth":1.5, "ecolor":"#333", "capthick":1.5})
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.08,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8.5)

    # Significance stars for PI vs RAG
    for j, r in enumerate(data):
        if r.get("pi_vs_rag_sig") == "Yes":
            y = max(safe_float(r.get("bm25_judge"),0),
                    safe_float(r.get("rag_judge"),0),
                    safe_float(r.get("pageindex_judge"),0)) + 0.35
            ax.text(x[j], y, "*", ha="center", fontsize=18, fontweight="bold", color="#222")

    ax.set_ylabel("LLM-as-Judge Score (0–4)", fontsize=12)
    ax.set_title("RQ1: Accuracy by Domain — BM25 vs Dense RAG vs PageIndex\n"
                 "Error bars = 95% bootstrap CI. ✱ p < 0.0125 (Bonferroni)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 5.2)
    ax.legend(fontsize=11, loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top","right"]: ax.spines[spine].set_visible(False)
    plt.tight_layout()
    _save(fig, "fig1_accuracy_by_domain")
    plt.close()


# ── Figure 2: Token cost ──────────────────────────────────────────────────────

def fig2_token_cost(rows):
    import matplotlib.pyplot as plt
    import numpy as np

    data   = [r for r in rows if r["domain"] != "OVERALL"
              and safe_float(r.get("bm25_tokens_mean")) is not None]
    labels = [SHORT.get(r["domain"], r["domain"]) for r in data]
    x = np.arange(len(labels))
    w = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (method, color, label) in enumerate(
        [("bm25","#E69F00","BM25"), ("rag","#0072B2","Dense RAG"), ("pageindex","#D55E00","PageIndex")]
    ):
        vals = [safe_float(r.get(f"{method}_tokens_mean"), 0) for r in data]
        ax.bar(x+(i-1)*w, vals, w, label=label, color=color, alpha=0.85)

    ax.set_ylabel("Mean Tokens per Question (Input + Output)", fontsize=12)
    ax.set_title("RQ3: Token Cost per Question by Domain", fontsize=12, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top","right"]: ax.spines[spine].set_visible(False)
    plt.tight_layout()
    _save(fig, "fig2_token_cost")
    plt.close()


# ── Figure 3: Accuracy gain vs cost ratio ────────────────────────────────────

def fig3_gain_vs_cost(acc_rows):
    import matplotlib.pyplot as plt

    data = [r for r in acc_rows if r["domain"] != "OVERALL"]
    xs, ys, labels = [], [], []
    for r in data:
        pi  = safe_float(r.get("pageindex_judge"))
        rag = safe_float(r.get("rag_judge"))
        pit = safe_float(r.get("pageindex_tokens_mean"))
        rat = safe_float(r.get("rag_tokens_mean"))
        if pi is None or rag is None or pit is None or rat is None or rat == 0:
            continue
        xs.append(pit/rat)
        ys.append(pi-rag)
        labels.append(SHORT.get(r["domain"], r["domain"]))

    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = ["#E69F00","#0072B2","#D55E00","#009E73"]
    for x, y, label, color in zip(xs, ys, labels, colors):
        ax.scatter(x, y, s=150, color=color, zorder=5, edgecolors="white", linewidths=0.8)
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(9,5), fontsize=11)

    ax.axhline(0, color="#999", linewidth=1, linestyle="--")
    ax.axvline(1, color="#999", linewidth=1, linestyle=":")

    ax.text(0.40, 0.97, "Better accuracy\nFewer tokens",
            transform=ax.transAxes, va="top", ha="center", fontsize=8, color="#2a6017", alpha=0.55,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8))
    ax.text(0.97, 0.03, "Worse accuracy\nMore tokens",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=9,
            color="#8b0000", alpha=0.75)

    ax.set_xlabel("Token Cost Ratio  (PageIndex / Dense RAG)", fontsize=12)
    ax.set_ylabel("Judge Score Gain  (PageIndex − Dense RAG)", fontsize=12)
    ax.set_title("RQ2+RQ3: Accuracy Gain vs Token Overhead",
                 fontsize=12, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.xaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    for spine in ["top","right"]: ax.spines[spine].set_visible(False)
    plt.tight_layout()
    _save(fig, "fig3_gain_vs_cost")
    plt.close()


# ── Figure 4: RQ2 — length tercile analysis ──────────────────────────────────

def fig4_length_analysis(rq2_rows):
    import matplotlib.pyplot as plt
    import numpy as np

    if not rq2_rows:
        print("  fig4: no RQ2 data, skipping")
        return

    labels = [SHORT.get(r["domain"], r["domain"]) for r in rq2_rows]
    short  = [safe_float(r.get("gap_short_docs"),  0) for r in rq2_rows]
    medium = [safe_float(r.get("gap_medium_docs"), 0) for r in rq2_rows]
    long   = [safe_float(r.get("gap_long_docs"),   0) for r in rq2_rows]

    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x-w,  short,  w, label="Short docs",  color="#56B4E9", alpha=0.85)
    ax.bar(x,    medium, w, label="Medium docs", color="#009E73", alpha=0.85)
    ax.bar(x+w,  long,   w, label="Long docs",   color="#CC79A7", alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Accuracy Gap (PageIndex Judge − RAG Judge)", fontsize=12)
    ax.set_title("RQ2: Does PageIndex Advantage Grow with Document Length?\n"
                 "Positive = PageIndex wins; Negative = RAG wins",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top","right"]: ax.spines[spine].set_visible(False)
    plt.tight_layout()
    _save(fig, "fig4_length_analysis")
    plt.close()


if __name__ == "__main__":
    print("=== Step 8: Generating Figures ===\n")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "matplotlib", "numpy", "-q"])

    acc_rows = load_csv(OUT / "accuracy_results.csv")
    rq2_rows = load_csv(OUT / "rq2_characteristics.csv")

    if not acc_rows:
        print("ERROR: Run script 07 first.")
        sys.exit(1)

    fig1_accuracy_bars(acc_rows)
    fig2_token_cost(acc_rows)
    fig3_gain_vs_cost(acc_rows)
    fig4_length_analysis(rq2_rows)

    print(f"\nAll figures → results/figures/")
    print("Step 8 complete.")
