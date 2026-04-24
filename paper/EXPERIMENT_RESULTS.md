# Experiment Results — RAG vs PageIndex

> Generated automatically from experimental data.

---

## Table 1: RQ1 — Accuracy Comparison (LLM-as-Judge, 0-4 scale)

| Domain | n | Dense RAG | PageIndex | PI vs RAG p | d | Sig? |
|---|---|---|---|---|---|
| Finance (FinanceBench) | 149 | 1.4161 [1.1342, 1.7047] | 0.8121 [0.5839, 1.0537] | 0.0008 | -0.2852 (small) | Yes |
| Legal (CUAD) | 149 | 0.5705 [0.4228, 0.7248] | 0.2013 [0.1074, 0.302] | 0.0 | -0.3521 (small) | Yes |
| Science (QASPER) | 150 | 0.3333 [0.2067, 0.48] | 0.6733 [0.48, 0.8667] | 0.0012 | 0.2482 (small) | Yes |
| Technology (SQuAD-Tech) | 149 | 3.4564 [3.2617, 3.6376] | 2.698 [2.4228, 2.9664] | 0.0 | -0.4296 (small) | Yes |
| OVERALL |  | 1.4422 [1.3032, 1.5829] | 1.0955 [0.9648, 1.2312] | 0.0 | -0.2069 (small) | Yes |

## Table 1b: Coverage-Adjusted (Answered Questions Only)

| Domain | Dense RAG (n/adj) | PageIndex (n/adj) | Gap |
|---|---|---|---|
| Finance (FinanceBench) | 73/149 / 2.73 | 45/149 / 2.56 | -0.17 |
| Legal (CUAD) | 54/149 / 1.46 | 14/149 / 1.71 | +0.25 |
| Science (QASPER) | 54/150 / 0.81 | 38/150 / 2.13 | +1.32 |
| Technology (SQuAD-Tech) | 137/149 / 3.76 | 109/149 / 3.69 | -0.07 |

## Table 2: Additional Accuracy Metrics

| Domain | Dense RAG F1 | PageIndex F1 | Dense RAG EM | PageIndex EM |
|---|---|---|---|---|
| Finance (FinanceBench) | 0.1765 | 0.1034 | 0.0134 | 0.0067 |
| Legal (CUAD) | 0.0845 | 0.0352 | 0 | 0 |
| Science (QASPER) | 0.1083 | 0.134 | 0 | 0 |
| Technology (SQuAD-Tech) | 0.7347 | 0.461 | 0.557 | 0.2685 |

## Table 3: RQ2 — Document Characteristics

| Domain | r(length,gap) | p | r(struct,gap) | p | Pattern |
|---|---|---|---|---|---|
| Finance (FinanceBench) | 0.0099 | 0.9042 | 0.0099 | 0.9042 | PI advantage grows with doc length |
| Legal (CUAD) | 0.4504 | 0.0 | 0.4504 | 0.0 | PI advantage grows with doc length |
| Science (QASPER) | 0.0509 | 0.5365 | 0.0509 | 0.5365 | PI advantage grows with doc length |
| Technology (SQuAD-Tech) | 0.1976 | 0.0157 | 0.1976 | 0.0157 | PI advantage grows with doc length |

## Table 4a: NOT FOUND Rates

| Domain | Dense RAG NF% | PageIndex NF% |
|---|---|---|
| Finance (FinanceBench) | 51.3% | 69.3% |
| Legal (CUAD) | 63.3% | 90.7% |
| Science (QASPER) | 64.0% | 74.7% |
| Technology (SQuAD-Tech) | 8.7% | 26.7% |

## Table 4b: Token Cost

| Domain | Dense RAG Tokens/Q | PageIndex Tokens/Q | PI:RAG Ratio |
|---|---|---|---|
| Finance (FinanceBench) | 1059 | 435 | 0.41x |
| Legal (CUAD) | 851 | 664 | 0.78x |
| Science (QASPER) | 620 | 781 | 1.26x |
| Technology (SQuAD-Tech) | 737 | 232 | 0.31x |

## Table 4c: Cost Efficiency (Tokens per Correct Answer)

| Domain | Dense RAG | PageIndex | Best |
|---|---|---|---|
| Finance (FinanceBench) | 3358 | 2315 | **PAGEINDEX** |
| Legal (CUAD) | 63507 | Inf | **RAG** |
| Science (QASPER) | 23221 | 7320 | **PAGEINDEX** |
| Technology (SQuAD-Tech) | 865 | 349 | **PAGEINDEX** |

---

## Statistical Notes

- Power analysis: For a paired Wilcoxon test with Cohen's d=0.3 (small-medium), α=0.05, and 1-β=0.80, required n=90 per domain. Our n=150 provides power ≈ 0.93, and n=600 overall provides power > 0.99.
- Paired Wilcoxon signed-rank test (non-parametric; judge scores are ordinal).
- Bonferroni correction applied for 4 simultaneous domain-level tests.
- Bootstrap 95% CIs computed with 10,000 resampling iterations (seed=42).

