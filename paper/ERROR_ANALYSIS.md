# Error Analysis -- RAG vs PageIndex Failure Modes

> Generated from LLM-classified error taxonomy (n=102, 10 per domain/method).

## Quantitative Summary

| Domain | PI Wins | RAG Wins | Ties |
|---|---|---|---|
| Finance (FinanceBench) | 92 | 3 | 55 |
| Legal (CUAD) | 34 | 9 | 107 |
| Science (QASPER) | 5 | 33 | 112 |
| Technology (SQuAD-Tech) | 0 | 82 | 68 |

## NOT FOUND Rate by System

PageIndex's 0% NOT FOUND rate on finance suggests it always generates an answer 
(even when retrieval may be imprecise), while RAG/BM25 honestly report when 
retrieved context lacks the answer.

| Domain | BM25 | RAG | PageIndex |
|---|---|---|---|
| Finance (FinanceBench) | 74% | 53% | 0% |
| Legal (CUAD) | 58% | 24% | 54% |
| Science (QASPER) | 43% | 48% | 74% |
| Technology (SQuAD-Tech) | 8% | 0% | 54% |

## Table 5: Failure Mode Taxonomy

| Category | Description | Count | % |
|---|---|---|---|
| ENTITY_LOOKUP | Failed to find/match specific entity | 71 | 69.6% |
| MULTI_HOP | Required multi-step reasoning | 6 | 5.9% |
| NUMERICAL | Involved calculations or numerical data | 17 | 16.7% |
| STRUCTURAL | Answer in specific doc structure (table, list) | 4 | 3.9% |
| LEXICAL_MATCH | Required exact terminology matching | 3 | 2.9% |
| LONG_RANGE | Context spread across distant sections | 1 | 1.0% |
| OTHER | Other failure modes | 0 | 0.0% |

## Failure Modes by System

| Category | RAG | BM25 | PageIndex |
|---|---|---|---|
| ENTITY_LOOKUP | 16 | 31 | 24 |
| MULTI_HOP | 3 | 3 | 0 |
| NUMERICAL | 8 | 6 | 3 |
| STRUCTURAL | 3 | 0 | 1 |
| LEXICAL_MATCH | 1 | 0 | 2 |
| LONG_RANGE | 1 | 0 | 0 |

## Representative Error Examples


### ENTITY_LOOKUP

- **financebench/bm25**: Q: Does Foot Locker's new CEO have previous CEO experience in a similar company to Footlocker?
  - Gold: Yes. She was previous CEO of Ulta Beauty which means she had to manage a large r
  - Pred: NOT FOUND
- **cuad/rag**: Q: Highlight the parts (if any) of this contract related to "Effective Date" that should be reviewed by
  - Gold: This contract shall be effective as of November 1, 2002, and shall remain in for
  - Pred: The effective date for purposes of the Agreement is the contract date as specifi

### MULTI_HOP

- **financebench/rag**: Q: Are Best Buy's gross margins historically consistent (not fluctuating more than roughly 2% each year
  - Gold: Yes, the margins have been consistent, there has been a minor decline of 1.1% in
  - Pred: NOT FOUND
- **financebench/rag**: Q: Basing your judgments off of the cash flow statement and the income statement, what is American Wate
  - Gold: $1832.00
  - Pred: NOT FOUND

### NUMERICAL

- **financebench/rag**: Q: What is FY2018 days payable outstanding (DPO) for Walmart? DPO is defined as: 365 * (average account
  - Gold: 42.69
  - Pred: NOT FOUND
- **financebench/rag**: Q: Does 3M have a reasonably healthy liquidity profile based on its quick ratio for Q2 of FY2023? If th
  - Gold: No. The quick ratio for 3M was 0.96 by Jun'23 close, which needs a bit of an imp
  - Pred: NOT FOUND

### STRUCTURAL

- **financebench/rag**: Q: By drawing conclusions from the information stated only in the income statement, what is Amazon's FY
  - Gold: $11588.00
  - Pred: NOT FOUND
- **financebench/rag**: Q: What is the quantity of restructuring costs directly outlined in Pepsico's income statements for FY2
  - Gold: Pepsico's restructuring costs in FY2022 amounted to $411 million .
  - Pred: NOT FOUND

### LEXICAL_MATCH

- **qasper/rag**: Q: How do they damage different neural modules?
  - Gold: Damage to neural modules is done by randomly initializing their weights, causing
  - Pred: NOT FOUND
- **qasper/pageindex**: Q: What learning paradigms do they cover in this survey?
  - Gold: Considering "What" and "How" separately versus jointly optimizing for both.
  - Pred: NOT FOUND

### LONG_RANGE

- **qasper/rag**: Q: how was the dataset built?
  - Gold: Questions are gathered from anonymized, aggregated queries to the Google search 
  - Pred: NOT FOUND
