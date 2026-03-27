"""
Script 09: Qualitative error analysis.

IEEE top-venue standard: quantitative results alone are not enough.
Reviewers expect analysis of *why* systems fail, not just *how much*.

This script:
  1. Finds cases where PageIndex clearly wins (PI judge ≥ 3, RAG judge ≤ 1)
  2. Finds cases where RAG wins (RAG judge ≥ 3, PI judge ≤ 1)
  3. Finds cases where BM25 surprisingly beats both
  4. Samples 10 from each category
  5. Uses LLM to categorize failure mode (multi-hop, entity lookup, numerical, etc.)
  6. Produces Table 5 for the paper: failure mode taxonomy

Output: paper/ERROR_ANALYSIS.md
"""

import json, random, sys
from pathlib import Path
from dotenv import load_dotenv
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from setup_copilot import get_client  # noqa

ROOT    = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")
SCORES  = ROOT / "results" / "scores"
PAPER   = ROOT / "paper"
PAPER.mkdir(exist_ok=True)

DOMAINS     = ["financebench", "cuad", "qasper", "techqa"]
DOMAIN_LABEL= {"financebench":"Finance","cuad":"Legal","qasper":"Science","techqa":"Tech"}
SEED        = 42
random.seed(SEED)
SAMPLE_SIZE = 10

CLASSIFY_PROMPT = """\
You are analyzing a question-answering failure to categorize the failure mode.

Question: {question}
Gold answer: {gold}
System A answer: {pred_a}
System A score: {score_a}/4
System B answer: {pred_b}
System B score: {score_b}/4

The system that scored LOWER failed. Categorize the failure into ONE of these types:

1. ENTITY_LOOKUP — question asks for a specific named entity, date, number
2. MULTI_HOP — answer requires connecting information from multiple passages
3. NUMERICAL — requires calculation or precise numerical extraction
4. STRUCTURAL — answer location depends on document structure (section headers, hierarchy)
5. LEXICAL_MATCH — answer uses exact terminology not paraphrased
6. LONG_RANGE — answer appears far from the semantically similar query passage
7. OTHER — none of the above

Reply with ONLY the category name (e.g., "MULTI_HOP"). Nothing else."""


def load_scores(domain: str, method: str) -> dict:
    p = SCORES / f"{domain}_{method}_scored.jsonl"
    if not p.exists():
        return {}
    return {json.loads(l)["id"]: json.loads(l) for l in open(p)
            if json.loads(l).get("judge_score", -1) >= 0}


def classify_failure(client, question, gold, pred_a, score_a, pred_b, score_b) -> str:
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",  # cheaper for classification
            messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(
                question=question, gold=gold,
                pred_a=pred_a, score_a=score_a,
                pred_b=pred_b, score_b=score_b,
            )}],
            max_tokens=20, temperature=0,
        )
        label = resp.choices[0].message.content.strip().upper()
        valid = {"ENTITY_LOOKUP","MULTI_HOP","NUMERICAL","STRUCTURAL",
                 "LEXICAL_MATCH","LONG_RANGE","OTHER"}
        return label if label in valid else "OTHER"
    except Exception:
        return "OTHER"


def analyze_domain(domain: str, client) -> dict:
    rag = load_scores(domain, "rag")
    pi  = load_scores(domain, "pageindex")
    bm25= load_scores(domain, "bm25")
    if not rag or not pi:
        return {}

    common = sorted(set(rag) & set(pi) & set(bm25)) if bm25 else sorted(set(rag) & set(pi))

    # Categorize cases
    pi_wins, rag_wins, bm25_wins, ties = [], [], [], []
    for qid in common:
        rj = rag[qid]["judge_score"]
        pj = pi[qid]["judge_score"]
        bj = bm25[qid]["judge_score"] if qid in bm25 else None

        if pj >= 3 and rj <= 1:
            pi_wins.append(qid)
        elif rj >= 3 and pj <= 1:
            rag_wins.append(qid)
        elif bj is not None and bj >= 3 and rj <= 1 and pj <= 1:
            bm25_wins.append(qid)
        elif pj == rj:
            ties.append(qid)

    print(f"  {domain}: PI wins={len(pi_wins)}  RAG wins={len(rag_wins)}  "
          f"BM25 surprises={len(bm25_wins)}  ties={len(ties)}")

    # Sample and classify
    results = {"pi_wins": [], "rag_wins": [], "bm25_surprises": []}
    for category, cases in [("pi_wins", pi_wins), ("rag_wins", rag_wins),
                              ("bm25_surprises", bm25_wins)]:
        sample = random.sample(cases, min(SAMPLE_SIZE, len(cases)))
        for qid in sample:
            r_row = rag[qid]
            p_row = pi[qid]
            failure_mode = classify_failure(
                client,
                r_row["question"], r_row["gold_answer"],
                r_row["pred_answer"], r_row["judge_score"],
                p_row["pred_answer"], p_row["judge_score"],
            )
            results[category].append({
                "id": qid, "domain": domain,
                "question": r_row["question"][:150],
                "gold": r_row["gold_answer"][:100],
                "rag_pred": r_row["pred_answer"][:100],
                "pi_pred":  p_row["pred_answer"][:100],
                "rag_score": r_row["judge_score"],
                "pi_score":  p_row["judge_score"],
                "failure_mode": failure_mode,
            })

    return {
        "domain":     domain,
        "pi_wins":    len(pi_wins),
        "rag_wins":   len(rag_wins),
        "bm25_wins":  len(bm25_wins),
        "ties":       len(ties),
        "classified": results,
    }


def write_error_analysis(domain_results: list):
    # Aggregate failure modes
    pi_win_modes  = Counter()
    rag_win_modes = Counter()
    for dr in domain_results:
        if "classified" not in dr:
            continue
        for r in dr["classified"]["pi_wins"]:
            pi_win_modes[r["failure_mode"]] += 1
        for r in dr["classified"]["rag_wins"]:
            rag_win_modes[r["failure_mode"]] += 1

    lines = [
        "# Error Analysis — Where Each System Wins and Fails",
        "",
        "## Quantitative Breakdown",
        "",
        "| Domain | PI wins | RAG wins | BM25 surprises | Ties |",
        "|---|---|---|---|---|",
    ]
    for dr in domain_results:
        if not dr:
            continue
        lines.append(
            f"| {DOMAIN_LABEL.get(dr['domain'], dr['domain'])} "
            f"| {dr.get('pi_wins',0)} | {dr.get('rag_wins',0)} "
            f"| {dr.get('bm25_wins',0)} | {dr.get('ties',0)} |"
        )

    lines += [
        "",
        "## Table 5: Failure Mode Taxonomy",
        "",
        "Question types where PageIndex wins vs where Dense RAG wins.",
        "",
        "| Failure Mode | PageIndex wins | RAG wins | Interpretation |",
        "|---|---|---|---|",
    ]
    all_modes = sorted(set(list(pi_win_modes.keys()) + list(rag_win_modes.keys())))
    mode_interp = {
        "ENTITY_LOOKUP": "RAG's semantic similarity finds named entity passages better",
        "MULTI_HOP":     "PageIndex tree enables cross-section reasoning",
        "NUMERICAL":     "Neither system handles calculation; similar failure rate",
        "STRUCTURAL":    "PageIndex tree captures document hierarchy advantage",
        "LEXICAL_MATCH": "BM25 and RAG benefit from exact term matching",
        "LONG_RANGE":    "PageIndex tree links distant passages; RAG top-k misses them",
        "OTHER":         "Miscellaneous; no clear pattern",
    }
    for mode in all_modes:
        lines.append(
            f"| {mode} | {pi_win_modes.get(mode,0)} | {rag_win_modes.get(mode,0)} "
            f"| {mode_interp.get(mode,'—')} |"
        )

    lines += [
        "",
        "## Representative Examples",
        "",
    ]
    for dr in domain_results:
        if not dr or "classified" not in dr:
            continue
        domain = DOMAIN_LABEL.get(dr["domain"], dr["domain"])
        for category, label in [("pi_wins","PageIndex wins"), ("rag_wins","RAG wins")]:
            for ex in dr["classified"][category][:2]:
                lines += [
                    f"**[{domain} — {label} — {ex['failure_mode']}]**",
                    f"- Q: {ex['question']}",
                    f"- Gold: {ex['gold']}",
                    f"- RAG ({ex['rag_score']}/4): {ex['rag_pred']}",
                    f"- PageIndex ({ex['pi_score']}/4): {ex['pi_pred']}",
                    "",
                ]

    out = PAPER / "ERROR_ANALYSIS.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"\n  Saved: {out.name}")


if __name__ == "__main__":
    print("=== Step 9: Error Analysis ===\n")
    client = get_client()
    domain_results = []
    for domain in DOMAINS:
        print(f"  Analyzing: {domain}")
        result = analyze_domain(domain, client)
        domain_results.append(result)

    write_error_analysis(domain_results)
    print("\nStep 9 complete.")
