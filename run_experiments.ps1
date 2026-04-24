# Run the complete RAG vs PageIndex evaluation pipeline.
#
# Due to GitHub Models API rate limits (150 req/day/model), this script
# uses different models per domain. Within each domain, both RAG and
# PageIndex use the SAME model for a fair comparison.
#
# All scripts are idempotent - they skip already-completed questions.
# You may need to run this multiple times across days if rate-limited.

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

Write-Host "`n=== Step 0: Refresh API Token ===" -ForegroundColor Cyan
python scripts/00_setup_copilot.py

Write-Host "`n=== Step 2: Build RAG Indexes (no API needed) ===" -ForegroundColor Cyan
python scripts/02_build_rag_index.py

Write-Host "`n=== Step 3b: Build PageIndex Trees (no API needed) ===" -ForegroundColor Cyan
python scripts/03b_build_pageindex_trees_official.py

Write-Host "`n=== Step 4: RAG Evaluation ===" -ForegroundColor Cyan
python scripts/04b_run_rag_multimodel.py

Write-Host "`n=== Step 5e: PageIndex Evaluation ===" -ForegroundColor Cyan
python scripts/05e_run_pageindex_official_eval.py

Write-Host "`n=== Step 6b: LLM Judge Scoring ===" -ForegroundColor Cyan
python scripts/06b_llm_judge_two_model.py

Write-Host "`n=== Step 7: Analysis ===" -ForegroundColor Cyan
python scripts/07_analyze_results.py

Write-Host "`n=== Complete ===" -ForegroundColor Green
Write-Host "Results in: results/"
Write-Host "Paper in: paper/PAPER_v4.md"
