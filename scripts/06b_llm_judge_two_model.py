"""
Script 06b: Proper LLM-as-Judge with TWO different models for reliability.

The original script was flawed - using the same model at temperature=0 twice
gives deterministic agreement, making Cohen's kappa meaningless.

This script implements proper inter-rater reliability:
- Primary judge: GPT-4o
- Secondary judge: GPT-4o-mini (different model for true reliability check)
- 20% holdout for reliability assessment
- Reports meaningful Cohen's kappa between the two different judges
"""

import json
import random
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from setup_copilot import get_client

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")
SCORES = ROOT / "results" / "scores"
SCORES.mkdir(parents=True, exist_ok=True)

DOMAINS = ["financebench", "cuad", "qasper", "techqa"]
METHODS = ["rag", "pageindex"]
METHOD_DIRS = {"rag": "rag_answers", "pageindex": "pageindex_answers"}

# Two different models for true inter-rater reliability
PRIMARY_JUDGE = "gpt-4o"
SECONDARY_JUDGE = "gpt-4o-mini"  # Different model = meaningful reliability

RELIABILITY_FRAC = 0.20  # 20% holdout for reliability
SEED = 42
random.seed(SEED)

JUDGE_PROMPT = """\
You are an expert evaluator for question-answering systems.

Score the predicted answer against the gold answer using this rubric:
4 = Fully correct and complete — contains all key information
3 = Mostly correct — minor omission or acceptable paraphrase
2 = Partially correct — captures main idea, missing key details
1 = Marginally relevant — barely addresses the question
0 = Wrong, irrelevant, hallucinated, or "NOT FOUND" when answer exists

Important:
- Award full credit for correct paraphrases
- Penalize hallucinated facts even if answer sounds plausible
- "NOT FOUND" scores 0 only if the answer is actually in the document

Question:    {question}
Gold answer: {gold}
Predicted:   {pred}

Reply with ONE integer (0, 1, 2, 3, or 4) and nothing else."""


def judge_one(client, question, gold, pred, model) -> int:
    """Score a single answer using specified judge model."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                question=question, gold=gold, pred=pred)}],
            max_tokens=5,
            temperature=0,
        )
        raw = resp.choices[0].message.content.strip()
        return max(0, min(4, int(raw[0])))
    except Exception as e:
        print(f"    Judge error: {e}")
        return -1


def score_domain_method(domain: str, method: str, client) -> Path:
    """Score all answers for a domain/method using primary judge."""
    src_path = ROOT / "results" / METHOD_DIRS[method] / f"{domain}.jsonl"
    out_path = SCORES / f"{domain}_{method}_scored.jsonl"

    if not src_path.exists():
        print(f"    SKIP: {src_path.name} not found")
        return out_path

    done_ids = {json.loads(l)["id"] for l in open(out_path)} if out_path.exists() else set()
    records = [json.loads(l) for l in open(src_path)]
    todo = [r for r in records if r["id"] not in done_ids]

    if not todo:
        rows = [json.loads(l) for l in open(out_path)]
        valid = [r for r in rows if r["judge_score"] >= 0]
        avg = sum(r["judge_score"] for r in valid) / len(valid) if valid else 0
        print(f"    {method:12s} [{domain}]: already scored (avg={avg:.3f}, n={len(valid)})")
        return out_path

    new_count = 0
    for r in todo:
        score = judge_one(client, r["question"], r["gold_answer"], r["pred_answer"], PRIMARY_JUDGE)
        with open(out_path, "a") as f:
            f.write(json.dumps({**r, "judge_score": score, "judge_model": PRIMARY_JUDGE}) + "\n")
        new_count += 1

    rows = [json.loads(l) for l in open(out_path)]
    valid = [r for r in rows if r["judge_score"] >= 0]
    avg = sum(r["judge_score"] for r in valid) / len(valid) if valid else 0
    print(f"    {method:12s} [{domain}]: +{new_count} scored  avg={avg:.3f}  n={len(valid)}")
    return out_path


def run_two_judge_reliability(client) -> dict:
    """
    Run proper inter-rater reliability with TWO different judge models.

    This provides meaningful reliability because:
    1. Two DIFFERENT models = independent judgments
    2. Disagreements reveal actual judgment variance
    3. Cohen's kappa measures agreement between different raters
    """
    print("\n=== Two-Judge Reliability Check ===")
    print(f"Primary judge: {PRIMARY_JUDGE}")
    print(f"Secondary judge: {SECONDARY_JUDGE}")
    print(f"Holdout fraction: {RELIABILITY_FRAC * 100:.0f}%\n")

    all_reliability = []

    for domain in DOMAINS:
        for method in METHODS:
            scored_path = SCORES / f"{domain}_{method}_scored.jsonl"
            if not scored_path.exists():
                continue

            rows = [json.loads(l) for l in open(scored_path)]
            valid = [r for r in rows if r["judge_score"] >= 0]
            if len(valid) < 10:
                continue

            # Sample holdout
            n_hold = max(10, int(len(valid) * RELIABILITY_FRAC))
            holdout = random.sample(valid, min(n_hold, len(valid)))

            # Get primary judge scores (already computed)
            scores_primary = [r["judge_score"] for r in holdout]

            # Get secondary judge scores (different model)
            scores_secondary = []
            for r in holdout:
                score = judge_one(client, r["question"], r["gold_answer"], r["pred_answer"], SECONDARY_JUDGE)
                scores_secondary.append(score if score >= 0 else 0)

            # Compute reliability metrics
            try:
                from scipy.stats import spearmanr
                r_val, p_val = spearmanr(scores_primary, scores_secondary)
            except Exception:
                r_val, p_val = float("nan"), float("nan")

            try:
                from sklearn.metrics import cohen_kappa_score
                kappa = cohen_kappa_score(scores_primary, scores_secondary, weights="quadratic")
            except Exception:
                kappa = float("nan")

            # Count exact agreements and disagreements
            exact_agree = sum(1 for a, b in zip(scores_primary, scores_secondary) if a == b)
            close_agree = sum(1 for a, b in zip(scores_primary, scores_secondary) if abs(a - b) <= 1)

            result = {
                "domain": domain,
                "method": method,
                "n_holdout": len(holdout),
                "primary_judge": PRIMARY_JUDGE,
                "secondary_judge": SECONDARY_JUDGE,
                "spearman_r": round(float(r_val), 4),
                "spearman_p": round(float(p_val), 4),
                "cohen_kappa_weighted": round(float(kappa), 4),
                "exact_agreement_pct": round(100 * exact_agree / len(holdout), 1),
                "close_agreement_pct": round(100 * close_agree / len(holdout), 1),
                "interpretation": (
                    "excellent" if kappa >= 0.8 else
                    "substantial" if kappa >= 0.6 else
                    "moderate" if kappa >= 0.4 else
                    "fair" if kappa >= 0.2 else
                    "poor"
                ),
            }
            all_reliability.append(result)

            print(f"  {domain}/{method}: κ={kappa:.3f} ({result['interpretation']}), "
                  f"r={r_val:.3f}, exact={exact_agree}/{len(holdout)}")

    # Summary statistics
    if all_reliability:
        avg_kappa = sum(r["cohen_kappa_weighted"] for r in all_reliability) / len(all_reliability)
        avg_r = sum(r["spearman_r"] for r in all_reliability) / len(all_reliability)
        print(f"\n  Overall: avg κ={avg_kappa:.3f}, avg r={avg_r:.3f}")
        print(f"  Note: κ between DIFFERENT models is the true reliability measure")

    return all_reliability


if __name__ == "__main__":
    print("=== Step 6b: Two-Judge LLM Scoring (Proper Reliability) ===\n")
    client = get_client()

    # Phase 1: Score all answers with primary judge
    print("Phase 1: Scoring all answers with primary judge...\n")
    for domain in DOMAINS:
        print(f"  Domain: {domain}")
        for method in METHODS:
            score_domain_method(domain, method, client)

    # Phase 2: Two-judge reliability
    reliability_results = run_two_judge_reliability(client)

    # Save reliability results
    rel_path = ROOT / "results" / "judge_reliability_two_model.json"
    with open(rel_path, "w") as f:
        json.dump(reliability_results, f, indent=2)
    print(f"\n  Two-judge reliability saved to: {rel_path.name}")

    # Summary
    print("\n--- Judge Score Summary ---")
    for domain in DOMAINS:
        row = [f"{domain}"]
        for method in METHODS:
            p = SCORES / f"{domain}_{method}_scored.jsonl"
            if not p.exists():
                row.append("—")
                continue
            rows = [json.loads(l) for l in open(p)]
            valid = [r for r in rows if r["judge_score"] >= 0]
            avg = sum(r["judge_score"] for r in valid) / len(valid) if valid else 0
            row.append(f"{avg:.2f}")
        print(f"  {row[0]:15s}  BM25={row[1]}  RAG={row[2]}  PageIndex={row[3]}")

    print("\nStep 6b complete.")
