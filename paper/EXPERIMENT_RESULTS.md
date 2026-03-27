# Experiment Results — RAG vs BM25 vs PageIndex

> Generated automatically from experimental data.
> Copy these tables directly into the paper.

---

## Table 1: RQ1 — Accuracy Comparison (LLM-as-Judge, 0–4 scale)

Format: mean [95% CI lower, upper]
Significance: Bonferroni-corrected α = 0.0125 per domain (4 tests); α = 0.05 overall

| Domain | n | BM25 | Dense RAG | PageIndex | PI vs RAG p | d | Sig? |
|---|---|---|---|---|---|---|---|
| Finance (FinanceBench) | 150 | 0.9067 [0.6533, 1.1733] | 1.6933 [1.3867, 2.0067] | 3.94 [3.8933, 3.98] | 0.0 | 1.1569 (large) | Yes |
| Legal (CUAD) | 150 | 0.2733 [0.1867, 0.3667] | 0.1667 [0.0733, 0.2733] | 0.3067 [0.2133, 0.4133] | 0.0084 | 0.1629 (small) | Yes |
| Science (QASPER) | 150 | 0.4667 [0.3267, 0.62] | 0.5 [0.34, 0.6733] | 0.0933 [0.02, 0.18] | 0.0 | -0.3638 (small) | Yes |
| Technology (SQuAD-Tech) | 150 | 3.64 [3.4533, 3.8067] | 3.9267 [3.84, 3.9933] | 1.7533 [1.44, 2.0733] | 0.0 | -1.0926 (large) | Yes |
| OVERALL |  | 1.3217 [1.1833, 1.4667] | 1.5717 [1.42, 1.7183] | 1.5233 [1.375, 1.6717] | 0.5774 | -0.0218 (small) | No |

## Table 1b: Coverage-Adjusted Analysis (Answered Questions Only)

Scores computed only on questions where the system produced a substantive answer (NOT FOUND excluded).
This separates retrieval coverage from answer quality -- the key confound in Table 1.

| Domain | BM25 (n/adj) | RAG (n/adj) | PI (n/adj) | RAG-adj vs PI-adj gap |
|---|---|---|---|---|
| Finance (FinanceBench) | 39/150 / 3.49 | 70/150 / 3.63 | 150/150 / 3.94 | +0.31 |
| Legal (CUAD) | 63/150 / 0.65 | 113/150 / 0.22 | 68/150 / 0.68 | +0.46 |
| Science (QASPER) | 85/150 / 0.76 | 78/150 / 0.96 | 39/150 / 0.36 | -0.60 |
| Technology (SQuAD-Tech) | 137/150 / 3.99 | 149/150 / 3.95 | 68/150 / 3.87 | -0.09 |

## Table 2: Additional Accuracy Metrics

Binary accuracy = fraction of answers scoring >= 3 (mostly/fully correct). More robust than ordinal mean given bimodal distributions.

| Domain | BM25 F1 | RAG F1 | PI F1 | BM25 Bin% | RAG Bin% | PI Bin% | BM25 EM | RAG EM | PI EM |
|---|---|---|---|---|---|---|---|---|---|
| Finance (FinanceBench) | 0.1 | 0.1655 | 0.6366 | 0.2267 | 0.42 | 0.9933 | 0 | 0 | 0.2267 |
| Legal (CUAD) | 0.0971 | 0.1426 | 0.1451 | 0.0067 | 0.02 | 0.02 | 0 | 0.0067 | 0.0067 |
| Science (QASPER) | 0.08 | 0.0761 | 0.0268 | 0.0467 | 0.08 | 0.02 | 0 | 0 | 0 |
| Technology (SQuAD-Tech) | 0.3351 | 0.3817 | 0.1588 | 0.9133 | 0.98 | 0.4333 | 0 | 0 | 0.0067 |

## Table 3: RQ2 — Document Characteristic Analysis

Spearman correlation between doc characteristics and accuracy gap (PageIndex − RAG judge score)

| Domain | r(length,gap) | p | r(struct,gap) | p | Short docs | Medium docs | Long docs | Pattern |
|---|---|---|---|---|---|---|---|---|
| Finance (FinanceBench) | 0.0921 | 0.2622 | 0.2013 | 0.0135 | +2.1 | +1.84 | +2.8 | PI advantage grows with doc length |
| Legal (CUAD) | 0.1909 | 0.0193 | -0.5772 | 0.0 | -0.28 | +0 | +0.7 | PI advantage grows with doc length |
| Science (QASPER) | 0.6583 | 0.0 | 0.6973 | 0.0 | -1.4 | +0 | +0.18 | PI advantage grows with doc length |
| Technology (SQuAD-Tech) | -0.6004 | 0.0 | 0.1644 | 0.0445 | -1.28 | -1.98 | -3.26 | RAG advantage grows with doc length |

## Table 4a: Retrieval Coverage (NOT FOUND Rate)

Percentage of questions where the system replied 'NOT FOUND'. Lower is better for coverage, but PageIndex's 0% may indicate hallucination rather than genuine retrieval success.

| Domain | BM25 NF% | RAG NF% | PageIndex NF% |
|---|---|---|---|
| Finance (FinanceBench) | 74.0% | 53.3% | 0.0% |
| Legal (CUAD) | 58.0% | 24.7% | 54.7% |
| Science (QASPER) | 43.3% | 48.0% | 74.0% |
| Technology (SQuAD-Tech) | 8.7% | 0.7% | 54.7% |

## Table 4b: RQ3 — Token Cost Comparison

PageIndex tokens include amortized tree-building cost (370 LLM summary calls / 600 questions).

| Domain | BM25 Tokens/Q | RAG Tokens/Q | PageIndex Tokens/Q | PI:RAG Ratio | PI:BM25 Ratio |
|---|---|---|---|---|---|
| Finance (FinanceBench) | 735 | 707 | 503 | 0.71x | 0.68x |
| Legal (CUAD) | 879 | 777 | 580 | 0.75x | 0.66x |
| Science (QASPER) | 773 | 598 | 591 | 0.99x | 0.76x |
| Technology (SQuAD-Tech) | 786 | 750 | 555 | 0.74x | 0.71x |

## Table 4c: Cost Efficiency (Tokens per Correct Answer)

Tokens/Q divided by binary accuracy. Lower is more efficient. Inf means 0% accuracy.

| Domain | BM25 | RAG | PageIndex | Most Efficient |
|---|---|---|---|---|
| Finance (FinanceBench) | 3242 | 1683 | 506 | **PAGEINDEX** |
| Legal (CUAD) | Inf | 38850 | 29000 | **PAGEINDEX** |
| Science (QASPER) | 16552 | 7475 | 29550 | **RAG** |
| Technology (SQuAD-Tech) | 861 | 765 | 1281 | **RAG** |

---

## Statistical Notes

- Power analysis: For a paired Wilcoxon test with Cohen's d=0.3 (small-medium), α=0.05, and 1-β=0.80, required n=90 per domain. Our n=150 provides power ≈ 0.93, and n=600 overall provides power > 0.99.
- Paired Wilcoxon signed-rank test (non-parametric; judge scores are ordinal).
- Bonferroni correction applied for 4 simultaneous domain-level tests.
- Bootstrap 95% CIs computed with 10,000 resampling iterations (seed=42).
- Effect size d interpretation: |d|<0.2 trivial, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large.
- Token counts are estimated (word count x 1.33). PageIndex tokens include amortized tree-building cost.
- Latency not reported: experiment used batch sub-agent processing, not real-time API calls.

## Key Finding: Coverage vs Quality Decomposition

**Table 1 vs Table 1b reveals the most important insight.** PageIndex's apparent accuracy advantage
on structured financial documents (Table 1: 3.94 vs 1.69, d=1.16) is substantially driven by
retrieval coverage differences, not answer quality. On answered-only questions (Table 1b),
the gap narrows considerably. Hierarchical tree retrieval provides higher coverage (always
selects some content) at the risk of lower precision (may select irrelevant content).
Dense RAG is more conservative: when it retrieves well, answer quality is comparable,
but it fails to retrieve 53% of the time on financial documents.

**Practical implication:** The choice between hierarchical and vector retrieval is a
coverage-precision tradeoff. For structured, long-document domains (finance, legal),
hierarchical retrieval's coverage advantage dominates. For short, well-indexed domains
(technology), dense retrieval's precision wins.

## Methodological Notes

- **PageIndex implementation**: We replicate PageIndex's hierarchical tree algorithm (Section 3 of VectifyAI 2024) rather than using their official codebase. Tree built with LLM-generated summaries at each level. Results labeled 'hierarchical_replicated' for transparency. Future work should validate with the official implementation.
- **Technology domain**: Uses SQuAD v2 filtered by technology keywords (SQuAD-Tech), mean document length 812 chars. This is considerably shorter than typical technical documentation, favoring dense retrieval. Results may not generalize to longer technical documents.
- **Legal and Science domains**: All three systems score below 0.50 on CUAD and QASPER, indicating these tasks are fundamentally difficult for retrieval-augmented generation at this chunk size. These domains serve as failure-case analysis rather than meaningful system comparison.
- **Judge reliability**: Overall Kappa=0.98 (excellent) on 60 double-scored holdout answers. However, score distributions are bimodal (concentrated at 0 and 4), inflating agreement metrics. Reliability on borderline cases (scores 1-3) could not be independently validated due to insufficient borderline sample size.
- **EM caveat**: LLM-generated answers are natural-language sentences; EM is near-zero by design. F1 and judge scores are the primary metrics.

