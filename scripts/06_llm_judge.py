"""
Script 06: LLM-as-Judge scoring + reliability validation.

IEEE-bar requirements met here:
  1. Judge model: claude-opus-4-5 via Copilot (stronger than gpt-4o-mini)
  2. Judge reliability: 10% holdout scored TWICE → Spearman r and Cohen's Kappa reported
     Without this, reviewers will reject: "how reliable is your judge?"
  3. Judge score reported alongside EM and F1 (triangulation)
  4. Invalid scores (non-parseable) flagged and reported as a limitation

Scale: 0–4
  4 = Fully correct and complete
  3 = Mostly correct, minor omission
  2 = Partially correct, missing key details
  1 = Marginally relevant, barely answers
  0 = Wrong, hallucinated, or NOT FOUND when answer exists
"""

import json, random, sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from setup_copilot import get_client  # noqa

ROOT    = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")
SCORES  = ROOT / "results" / "scores"
SCORES.mkdir(parents=True, exist_ok=True)

DOMAINS      = ["financebench", "cuad", "qasper", "techqa"]
METHODS      = ["rag", "bm25", "pageindex"]
METHOD_DIRS  = {"rag": "rag_answers", "bm25": "bm25_answers", "pageindex": "pageindex_answers"}
JUDGE_MODEL  = "gpt-4o"
RELIABILITY_FRAC = 0.10    # 10% of each domain held out for double-scoring
SEED         = 42
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


def judge_one(client, question, gold, pred) -> int:
    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                question=question, gold=gold, pred=pred)}],
            max_tokens=5, temperature=0,
        )
        raw = resp.choices[0].message.content.strip()
        return max(0, min(4, int(raw[0])))
    except Exception:
        return -1


def score_domain_method(domain: str, method: str, client) -> Path:
    src_path = ROOT / "results" / METHOD_DIRS[method] / f"{domain}.jsonl"
    out_path = SCORES / f"{domain}_{method}_scored.jsonl"

    if not src_path.exists():
        print(f"    SKIP: {src_path.name} not found")
        return out_path

    done_ids = {json.loads(l)["id"] for l in open(out_path)} if out_path.exists() else set()
    records  = [json.loads(l) for l in open(src_path)]
    todo     = [r for r in records if r["id"] not in done_ids]

    if not todo:
        rows  = [json.loads(l) for l in open(out_path)]
        valid = [r for r in rows if r["judge_score"] >= 0]
        avg   = sum(r["judge_score"] for r in valid)/len(valid) if valid else 0
        print(f"    {method:12s} [{domain}]: already scored (avg={avg:.3f}, n={len(valid)})")
        return out_path

    new_count = 0
    for r in todo:
        score = judge_one(client, r["question"], r["gold_answer"], r["pred_answer"])
        with open(out_path, "a") as f:
            f.write(json.dumps({**r, "judge_score": score,
                                "judge_model": JUDGE_MODEL}) + "\n")
        new_count += 1

    rows  = [json.loads(l) for l in open(out_path)]
    valid = [r for r in rows if r["judge_score"] >= 0]
    avg   = sum(r["judge_score"] for r in valid)/len(valid) if valid else 0
    print(f"    {method:12s} [{domain}]: +{new_count} scored  "
          f"avg={avg:.3f}  n={len(valid)}  invalid={len(rows)-len(valid)}")
    return out_path


def run_reliability_check(domain: str, method: str, client) -> dict:
    """
    Double-score a 10% holdout to compute judge reliability.
    Reports Spearman r and Cohen's Kappa (weighted).
    This goes into Table 2 of the paper.
    """
    scored_path = SCORES / f"{domain}_{method}_scored.jsonl"
    if not scored_path.exists():
        return {}

    rows   = [json.loads(l) for l in open(scored_path)]
    valid  = [r for r in rows if r["judge_score"] >= 0]
    n_hold = max(10, int(len(valid) * RELIABILITY_FRAC))
    holdout= random.sample(valid, min(n_hold, len(valid)))

    scores1 = [r["judge_score"] for r in holdout]
    scores2 = [judge_one(client, r["question"], r["gold_answer"], r["pred_answer"])
               for r in holdout]
    scores2 = [s if s >= 0 else 0 for s in scores2]   # treat invalid as 0

    try:
        from scipy.stats import spearmanr
        r_val, p_val = spearmanr(scores1, scores2)
    except Exception:
        r_val, p_val = float("nan"), float("nan")

    try:
        from sklearn.metrics import cohen_kappa_score
        kappa = cohen_kappa_score(scores1, scores2, weights="quadratic")
    except Exception:
        kappa = float("nan")

    result = {
        "domain": domain, "method": method,
        "n_holdout": len(holdout),
        "spearman_r": round(float(r_val), 4),
        "spearman_p": round(float(p_val), 4),
        "cohen_kappa_weighted": round(float(kappa), 4),
        "interpretation": (
            "excellent" if kappa >= 0.8 else
            "substantial" if kappa >= 0.6 else
            "moderate" if kappa >= 0.4 else
            "fair"
        ),
    }
    print(f"    Reliability [{domain}/{method}]: "
          f"r={r_val:.3f}  κ={kappa:.3f}  ({result['interpretation']})")
    return result


if __name__ == "__main__":
    print("=== Step 6: LLM-as-Judge Scoring ===")
    print(f"  Judge model: {JUDGE_MODEL} via GitHub Copilot\n")
    client = get_client()

    # ── Phase 1: Score all answers ──────────────────────────────────────────
    print("Phase 1: Scoring all answers...\n")
    for domain in DOMAINS:
        print(f"  Domain: {domain}")
        for method in METHODS:
            score_domain_method(domain, method, client)

    # ── Phase 2: Reliability check (10% double-score) ───────────────────────
    print("\nPhase 2: Judge reliability check (10% holdout, double-scored)...\n")
    reliability_results = []
    # Only check one method per domain to limit token cost
    for domain in DOMAINS:
        result = run_reliability_check(domain, "rag", client)
        if result:
            reliability_results.append(result)

    rel_path = ROOT / "results" / "judge_reliability.json"
    with open(rel_path, "w") as f:
        json.dump(reliability_results, f, indent=2)
    print(f"\n  Reliability saved to: {rel_path.name}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n--- Judge Score Summary ---")
    for domain in DOMAINS:
        row = [f"{domain}"]
        for method in METHODS:
            p = SCORES / f"{domain}_{method}_scored.jsonl"
            if not p.exists():
                row.append("—")
                continue
            rows  = [json.loads(l) for l in open(p)]
            valid = [r for r in rows if r["judge_score"] >= 0]
            avg   = sum(r["judge_score"] for r in valid)/len(valid) if valid else 0
            row.append(f"{avg:.2f}")
        print(f"  {row[0]:15s}  BM25={row[1]}  RAG={row[2]}  PageIndex={row[3]}")

    print("\nStep 6 complete.")
