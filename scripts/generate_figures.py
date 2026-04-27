"""Generate publication-quality figures from experiment results."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

DOMAINS = ["financebench", "cuad", "qasper", "techqa"]
LABELS = {"financebench": "Finance", "cuad": "Legal", "qasper": "Science", "techqa": "Technology"}

def load_answers():
    data = {}
    for d in DOMAINS:
        data[d] = {
            "rag": [json.loads(l) for l in open(ROOT / f"results/rag_answers/{d}.jsonl")],
            "pi": [json.loads(l) for l in open(ROOT / f"results/pageindex_answers/{d}.jsonl")],
        }
    return data

def load_scores():
    data = {}
    for d in DOMAINS:
        for m in ["rag", "pageindex"]:
            rows = [json.loads(l) for l in open(ROOT / f"results/scores/{d}_{m}_scored.jsonl")]
            valid = [r for r in rows if r.get("judge_score", -1) >= 0]
            data[(d, m)] = valid
    return data


def fig1_accuracy_by_domain(scores):
    """Bar chart: Judge scores by domain, RAG vs PageIndex."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(DOMAINS))
    width = 0.35

    rag_scores = [np.mean([r["judge_score"] for r in scores[(d, "rag")]]) for d in DOMAINS]
    pi_scores = [np.mean([r["judge_score"] for r in scores[(d, "pageindex")]]) for d in DOMAINS]

    bars1 = ax.bar(x - width/2, rag_scores, width, label="Dense RAG", color="#4C72B0", edgecolor="white")
    bars2 = ax.bar(x + width/2, pi_scores, width, label="PageIndex", color="#DD8452", edgecolor="white")

    ax.set_ylabel("Judge Score (0-4)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[d] for d in DOMAINS], fontsize=11)
    ax.legend(fontsize=11, loc="upper left")
    ax.set_ylim(0, 4.2)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("RQ1: Accuracy by Domain (LLM-as-Judge, 0-4 scale)", fontsize=13)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIG / f"fig1_accuracy_by_domain.{ext}", dpi=300)
    plt.close()
    print("  fig1_accuracy_by_domain saved")


def fig2_not_found_rates(answers):
    """Bar chart: NOT FOUND rates by domain."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(DOMAINS))
    width = 0.35

    rag_nf = [100 * sum(1 for r in answers[d]["rag"] if "NOT FOUND" in r["pred_answer"].upper()) / len(answers[d]["rag"]) for d in DOMAINS]
    pi_nf = [100 * sum(1 for r in answers[d]["pi"] if "NOT FOUND" in r["pred_answer"].upper()) / len(answers[d]["pi"]) for d in DOMAINS]

    bars1 = ax.bar(x - width/2, rag_nf, width, label="Dense RAG", color="#4C72B0", edgecolor="white")
    bars2 = ax.bar(x + width/2, pi_nf, width, label="PageIndex", color="#DD8452", edgecolor="white")

    ax.set_ylabel("NOT FOUND Rate (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[d] for d in DOMAINS], fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Retrieval Coverage: NOT FOUND Rates", fontsize=13)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIG / f"fig2_not_found_rates.{ext}", dpi=300)
    plt.close()
    print("  fig2_not_found_rates saved")


def fig3_both_found_f1(answers):
    """Bar chart: F1 on both-found subsets."""
    fig, ax = plt.subplots(figsize=(8, 5))

    domains_with_data = []
    rag_f1s = []
    pi_f1s = []
    ns = []

    for d in DOMAINS:
        pi_ids = {r["id"]: r for r in answers[d]["pi"]}
        both = [(r, pi_ids[r["id"]]) for r in answers[d]["rag"]
                if r["id"] in pi_ids
                and "NOT FOUND" not in r["pred_answer"].upper()
                and "NOT FOUND" not in pi_ids[r["id"]]["pred_answer"].upper()]
        if len(both) >= 5:
            domains_with_data.append(d)
            rag_f1s.append(np.mean([r["f1"] for r, _ in both]))
            pi_f1s.append(np.mean([p["f1"] for _, p in both]))
            ns.append(len(both))

    x = np.arange(len(domains_with_data))
    width = 0.35

    bars1 = ax.bar(x - width/2, rag_f1s, width, label="Dense RAG", color="#4C72B0", edgecolor="white")
    bars2 = ax.bar(x + width/2, pi_f1s, width, label="PageIndex", color="#DD8452", edgecolor="white")

    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{LABELS[d]}\n(n={n})" for d, n in zip(domains_with_data, ns)], fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Answer Quality: Both-Found Subset (F1)", fontsize=13)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIG / f"fig3_both_found_f1.{ext}", dpi=300)
    plt.close()
    print("  fig3_both_found_f1 saved")


def fig4_token_cost(answers):
    """Bar chart: Token cost per question."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(DOMAINS))
    width = 0.35

    rag_tok = [np.mean([r.get("tokens_in", 0) for r in answers[d]["rag"]]) for d in DOMAINS]
    pi_tok = [np.mean([r.get("tokens_in", 0) for r in answers[d]["pi"]]) for d in DOMAINS]

    bars1 = ax.bar(x - width/2, rag_tok, width, label="Dense RAG", color="#4C72B0", edgecolor="white")
    bars2 = ax.bar(x + width/2, pi_tok, width, label="PageIndex", color="#DD8452", edgecolor="white")

    ax.set_ylabel("Tokens per Question", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[d] for d in DOMAINS], fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("RQ3: Token Cost per Question", fontsize=13)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIG / f"fig4_token_cost.{ext}", dpi=300)
    plt.close()
    print("  fig4_token_cost saved")


if __name__ == "__main__":
    print("=== Generating Figures ===")
    answers = load_answers()
    scores = load_scores()

    fig1_accuracy_by_domain(scores)
    fig2_not_found_rates(answers)
    fig3_both_found_f1(answers)
    fig4_token_cost(answers)

    print(f"\nFigures saved to: {FIG}")
    for f in sorted(FIG.glob("*")):
        print(f"  {f.name}")
