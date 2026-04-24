#!/usr/bin/env bash
# Run the complete RAG vs PageIndex evaluation pipeline.
# All scripts are idempotent - they skip already-completed questions.

set -e
cd "$(dirname "$0")"

echo "=== Step 0: Refresh API Token ==="
python3 scripts/00_setup_copilot.py || python scripts/00_setup_copilot.py

echo ""
echo "=== Step 2: Build RAG Indexes (no API needed) ==="
python3 scripts/02_build_rag_index.py || python scripts/02_build_rag_index.py

echo ""
echo "=== Step 3: Build PageIndex Trees (no API needed) ==="
python3 scripts/03b_build_pageindex_trees_official.py || python scripts/03b_build_pageindex_trees_official.py

echo ""
echo "=== Step 4: RAG Evaluation ==="
python3 scripts/04b_run_rag_multimodel.py || python scripts/04b_run_rag_multimodel.py

echo ""
echo "=== Step 5: PageIndex Evaluation ==="
python3 scripts/05e_run_pageindex_official_eval.py || python scripts/05e_run_pageindex_official_eval.py

echo ""
echo "=== Step 6: LLM Judge Scoring ==="
python3 scripts/06b_llm_judge_two_model.py || python scripts/06b_llm_judge_two_model.py

echo ""
echo "=== Step 7: Analysis ==="
python3 scripts/07_analyze_results.py || python scripts/07_analyze_results.py

echo ""
echo "=== Complete ==="
echo "Results in: results/"
echo "Paper in: paper/PAPER.md"