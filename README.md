# Does Hierarchical Retrieval Outperform Vector Search on Evidence Passages?

**A Controlled Comparison of RAG and PageIndex across Four Document QA Domains**

Shivam Rawat -- Microsoft Azure

---

## Key Results

We compare Dense RAG (FAISS + all-MiniLM-L6-v2) against a reimplementation of [PageIndex](https://github.com/VectifyAI/PageIndex) (VectifyAI's tree-based retrieval) on 600 questions across 4 domains. Both systems use the same LLM within each domain.

| Domain | System | NOT FOUND % | F1 | Judge (0-4) |
|--------|--------|------------|-----|-------------|
| Finance | Dense RAG | 51.3% | 0.175 | 1.42 |
| Finance | PageIndex | 69.3% | 0.104 | 0.83 |
| Legal | Dense RAG | 63.3% | 0.085 | 0.57 |
| Legal | PageIndex | 90.7% | 0.035 | 0.20 |
| Science | Dense RAG | 64.0% | 0.108 | 0.33 |
| Science | PageIndex | 74.7% | 0.134 | **0.67** |
| Technology | Dense RAG | 8.7% | 0.730 | 3.43 |
| Technology | PageIndex | 26.7% | 0.465 | 2.70 |

**Finding:** PageIndex has higher NOT FOUND rates than RAG on all 4 domains. This is consistent with a mismatch between PageIndex's design (long, natively structured documents) and our evaluation data (short evidence passages with paragraph-derived structure). A context asymmetry (RAG retrieves ~3,500 chars vs. PageIndex ~1,500 chars) is a contributing confound.

On technology (n=105 both-found pairs): RAG F1=0.781 vs. PageIndex F1=0.636 (p<0.001, 95% CI [+0.084, +0.206]).

## Quick Start

### Prerequisites

- Python 3.10+
- [GitHub CLI](https://cli.github.com/) authenticated (`gh auth login`)
- ~2GB disk space

### Setup

```bash
git clone https://github.com/shivam2407/rag-vs-pageindex.git
cd rag-vs-pageindex
pip install -r requirements.txt
```

### Reproduce Results

The repository includes all 1,200 pre-computed answers and judge scores. You can verify them directly:

```bash
python scripts/07_analyze_results.py
```

### Re-run Experiments from Scratch

To re-run the full pipeline (requires GitHub Models API access):

```powershell
# Windows (PowerShell)
.\run_experiments.ps1

# Linux/macOS
bash run_experiments.sh
```

**Note:** The GitHub Models API has rate limits (150 requests/day/model for free-tier accounts). The pipeline is **idempotent** -- it skips already-completed questions, so you can run it multiple times across days.

## Pipeline

The experiment runs in 7 steps. Each step is a standalone script that can be run independently.

| Step | Script | What it does | API needed? |
|------|--------|-------------|-------------|
| 0 | `00_setup_copilot.py` | Gets GitHub API token via `gh auth token` | No |
| 1 | `01_download_datasets.py` | Downloads 150 QA pairs per domain from HuggingFace | No |
| 2 | `02_build_rag_index.py` | Builds FAISS vector index per domain (all-MiniLM-L6-v2) | No |
| 3 | `03b_build_pageindex_trees_official.py` | Builds hierarchical trees using VectifyAI's parsing algorithm | No |
| 4 | `04b_run_rag_multimodel.py` | Runs RAG evaluation (150 Q per domain) | Yes |
| 5 | `05e_run_pageindex_official_eval.py` | Runs PageIndex evaluation with tree navigation | Yes |
| 6 | `06b_llm_judge_two_model.py` | LLM-as-judge scoring (GPT-4o) + two-model reliability | Yes |
| 7 | `07_analyze_results.py` | Generates accuracy tables, statistics, coverage analysis | No |

### Generation Models (per domain)

Due to API rate limits, each domain uses a different generation model. **Within each domain, both RAG and PageIndex use the same model** for a controlled comparison.

| Domain | Model | Rationale |
|--------|-------|-----------|
| Finance | gpt-4.1-mini | Strong general-purpose |
| Legal | gpt-4.1-nano | Lightweight, fast |
| Science | Phi-4 | Microsoft's 14B model |
| Technology | Cohere-command-r | Strong instruction-following |

Cross-domain comparisons should be interpreted cautiously due to different models.

## Datasets

| Domain | Source | Questions | Avg. Length | Structure |
|--------|--------|-----------|-------------|-----------|
| Finance | [FinanceBench](https://huggingface.co/datasets/PatronusAI/financebench) | 150 | 1,284 chars | Low (0.00) |
| Legal | [CUAD](https://huggingface.co/datasets/cuad) | 150 | 4,958 chars | Low (0.10) |
| Science | [QASPER](https://huggingface.co/datasets/allenai/qasper) | 150 | 7,918 chars | High (0.98) |
| Technology | SQuAD-Tech | 150 | 812 chars | Low (0.00) |

**Important:** These are pre-extracted **evidence passages** (800-8,000 chars), not full documents. PageIndex was designed for long, multi-page documents with native section headers. Our results characterize performance on shorter passages and likely understate PageIndex's advantage on its intended use case.

### Data Cleaning

FinanceBench evidence passages were originally stored as Python dictionary representations (`{'evidence_text': '...', 'doc_name': '...'}`). We cleaned all 150 records by parsing the dictionary and extracting raw evidence text. The NOT FOUND rate on clean data (51.3%) matches the corrupted-data rate (53.3%), confirming retrieval failures are a genuine embedding limitation, not a data artifact.

## Methodology

### Dense RAG
- **Embedding:** all-MiniLM-L6-v2 (384-dim, local inference)
- **Index:** FAISS cosine similarity
- **Retrieval:** Top-5 chunks (chunk_size=1000, overlap=200), concatenated up to 3,500 chars

### PageIndex (Reimplementation)
- **Tree building:** Reimplements VectifyAI's `extract_nodes_from_markdown` and `build_tree_from_nodes` from their [open-source codebase](https://github.com/VectifyAI/PageIndex)
- **Section boundaries:** For passages without markdown headers, paragraph-derived sections are created
- **Navigation:** LLM sees tree structure (titles + hierarchy, no text) -> selects nodes -> reads selected node text (~1,500 chars)
- **This is NOT VectifyAI's full system** -- it is a faithful reimplementation of their parsing and navigation algorithm

### Evaluation
- **Primary metrics:** Token-level F1, Exact Match (EM)
- **LLM-as-judge:** GPT-4o scores on 0-4 scale
- **Judge reliability:** Two-model check (GPT-4o primary, GPT-4o-mini secondary). Weighted Cohen's kappa: 0.698-0.953 across domains (substantial to excellent)
- **Statistical tests:** Paired Wilcoxon signed-rank with bootstrap 95% CIs on both-found subsets

### Known Confounds (Disclosed)
1. **Context asymmetry:** RAG retrieves ~3,500 chars; PageIndex retrieves ~1,500 chars per query
2. **Paragraph-derived sections:** Automatically generated headers provide weak semantic signal for navigation
3. **Different models across domains:** Within-domain comparisons are controlled; cross-domain are not
4. **Same-vendor judges:** Both GPT-4o and GPT-4o-mini are from OpenAI

## Repository Structure

```
.
├── scripts/
│   ├── 00_setup_copilot.py            # GitHub API token setup
│   ├── setup_copilot.py               # Import helper for other scripts
│   ├── 01_download_datasets.py        # Download QA pairs from HuggingFace
│   ├── 02_build_rag_index.py          # Build FAISS vector indexes
│   ├── 03b_build_pageindex_trees_official.py  # Build PageIndex trees (VectifyAI algorithm)
│   ├── 04b_run_rag_multimodel.py      # RAG evaluation (multi-model)
│   ├── 05e_run_pageindex_official_eval.py     # PageIndex evaluation (tree navigation)
│   ├── 06b_llm_judge_two_model.py     # LLM-as-judge + reliability
│   ├── 07_analyze_results.py          # Results analysis
│   ├── 08_generate_figures.py         # Figure generation
│   └── fix_01_clean_financebench_data.py      # FinanceBench data cleaner
├── data/
│   ├── financebench/qa_pairs.jsonl    # 150 finance QA pairs (cleaned)
│   ├── cuad/qa_pairs.jsonl            # 150 legal QA pairs
│   ├── qasper/qa_pairs.jsonl          # 150 science QA pairs
│   └── techqa/qa_pairs.jsonl          # 150 technology QA pairs
├── results/
│   ├── rag_answers/                   # 600 RAG answers (4 × 150)
│   ├── pageindex_answers/             # 600 PageIndex answers (4 × 150)
│   ├── scores/                        # 1,200 judge scores (8 files)
│   └── judge_reliability_two_model.json
├── paper/
│   └── PAPER.md                       # Full paper manuscript
├── requirements.txt
├── run_experiments.ps1                # Windows runner
├── run_experiments.sh                 # Linux/macOS runner
└── .gitignore
```

### Generated at Runtime (not in repo)

These are generated by the pipeline and excluded from git (large / reproducible):

- `results/rag_index/` -- FAISS pickle files (~360MB). Generated by Step 2.
- `results/pageindex_trees/` -- 456 JSON tree files (~2.4MB). Generated by Step 3.
- `vendor/PageIndex/` -- VectifyAI clone. Referenced by Step 3 for algorithm validation.

## Verifying Results

### Quick Verification

```bash
# Check answer counts
python -c "
import json
for d in ['financebench','cuad','qasper','techqa']:
    rag = len(list(open(f'results/rag_answers/{d}.jsonl')))
    pi = len(list(open(f'results/pageindex_answers/{d}.jsonl')))
    print(f'{d}: RAG={rag} PI={pi}')
"
```

Expected output:
```
financebench: RAG=150 PI=150
cuad: RAG=150 PI=150
qasper: RAG=150 PI=150
techqa: RAG=150 PI=150
```

### Full Analysis

```bash
python scripts/07_analyze_results.py
```

### Re-run Judge Scoring Only

```bash
python scripts/00_setup_copilot.py
python scripts/06b_llm_judge_two_model.py
```

## Addressing Reviewer Feedback

This study was developed in response to specific reviewer concerns about an earlier version:

| Concern | Resolution |
|---------|------------|
| FinanceBench data corrupted (dict repr strings) | Cleaned. NOT FOUND rate persists at 51.3% (was 53.3%), proving it's a real embedding limitation |
| Documents framed as "200-page reports" | Reframed as "evidence passages" with explicit scope limitation |
| PageIndex was fake (manual 3-level tree) | Reimplemented using VectifyAI's parsing algorithm. Honestly labeled as reimplementation |
| Cohen's kappa meaningless (same model at temp=0) | Two-model judge (GPT-4o + GPT-4o-mini). Kappa: 0.698-0.953 |
| Model confound (different models between systems) | Same model within each domain. Different models across domains (disclosed) |

## Citation

```bibtex
@article{rawat2025ragpageindex,
  title={Does Hierarchical Retrieval Outperform Vector Search on Evidence Passages? A Controlled Comparison of RAG and PageIndex},
  author={Rawat, Shivam},
  year={2025}
}
```

## License

MIT
