# When Does Hierarchical Retrieval Beat Vector Search?

A cross-domain empirical study comparing BM25, dense RAG, and hierarchical tree retrieval for document question answering.

**Paper**: [PDF](paper/latex/paper.pdf) | [Google Drive](https://drive.google.com/file/d/16yDXBJiq6jFQszVTToa46B2yDU_KT3iu/view?usp=sharing)

## Key Finding

Accuracy differences between hierarchical and vector retrieval are mostly **retrieval coverage artifacts**, not answer quality differences. When both systems find relevant content, they answer about equally well.

| Domain | Dense RAG | Tree Retrieval | What explains the gap |
|--------|-----------|---------------|----------------------|
| Finance | 1.69 | **3.94** | RAG says "NOT FOUND" 53% of the time |
| Technology | **3.93** | 1.75 | Tree retrieval says "NOT FOUND" 55% of the time |
| Overall | 1.57 | 1.52 | No significant difference (p=0.58) |

## Experiment

- **3 systems**: BM25, Dense RAG (FAISS + all-MiniLM-L6-v2), Hierarchical Tree Retrieval (PageIndex algorithm)
- **4 domains**: Finance (FinanceBench), Legal (CUAD), Science (QASPER), Technology (SQuAD-Tech)
- **600 questions** (150 per domain), same LLM backbone for all systems
- **1,800 LLM-as-judge evaluations** with reliability validation (Cohen's kappa = 0.98)

## Repository Structure

```
scripts/              # Experiment pipeline (steps 00-09)
data/                 # 600 QA pairs across 4 domains
results/
  accuracy_results.csv      # Main results table
  rq2_characteristics.csv   # Document characteristic analysis
  judge_reliability.json    # Judge reliability metrics
  figures/                  # Publication figures (PNG + PDF)
paper/
  latex/paper.pdf           # IEEE-format paper
  EXPERIMENT_RESULTS.md     # All tables for the paper
  ERROR_ANALYSIS.md         # Failure mode taxonomy
```

## Reproducing the Experiment

```bash
pip install -r requirements.txt
python scripts/01_download_datasets.py
python scripts/02_build_indexes.py
# Steps 03-06 require LLM access (see scripts for details)
python scripts/07_analyze_results.py
python scripts/08_generate_figures.py
```

## Citation

```bibtex
@article{rawat2026hierarchical,
  title={When Does Hierarchical Retrieval Beat Vector Search? A Cross-Domain Empirical Study},
  author={Rawat, Shivam},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE).
