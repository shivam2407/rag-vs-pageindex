"""
Fix script: Clean corrupted FinanceBench context_text data.

Issue: The original download script stored context_text as str(evidence_dict),
resulting in Python dict repr strings like "{'evidence_text': '...' \\n ...}".

Fix: Parse the dict string and extract the actual evidence_text content.
Also properly convert escaped \\n back to real newlines.
"""

import json, ast, re
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "financebench"
QA_FILE = DATA / "qa_pairs.jsonl"
BACKUP_FILE = DATA / "qa_pairs_corrupted_backup.jsonl"


def parse_corrupted_context(context_text: str) -> str:
    """
    Parse a corrupted context_text field that contains a Python dict repr.
    Returns the cleaned evidence text with proper newlines.
    """
    if not context_text:
        return ""

    # If it doesn't look like a dict repr, return as-is
    if not context_text.strip().startswith("{"):
        return context_text

    # Try to parse as Python literal
    try:
        # The string may contain multiple concatenated dicts
        # Try to extract just the first evidence_text
        data = ast.literal_eval(context_text)
        if isinstance(data, dict):
            text = data.get("evidence_text", "")
            if not text:
                text = data.get("evidence_text_full_page", "")
            # Convert escaped newlines to real newlines
            text = text.replace("\\n", "\n")
            return text.strip()
    except (ValueError, SyntaxError):
        pass

    # Fallback: regex extraction
    # Look for 'evidence_text': '...' pattern
    match = re.search(r"'evidence_text':\s*'(.*?)'(?:,\s*'|\s*})", context_text, re.DOTALL)
    if match:
        text = match.group(1)
        # Unescape
        text = text.replace("\\n", "\n").replace("\\'", "'")
        return text.strip()

    # More aggressive fallback: find the longest quoted string
    matches = re.findall(r"'([^']{100,})'", context_text)
    if matches:
        longest = max(matches, key=len)
        return longest.replace("\\n", "\n").strip()

    # Last resort: strip the dict-like parts
    cleaned = context_text
    cleaned = re.sub(r"^\s*\{?\s*'evidence_text':\s*'", "", cleaned)
    cleaned = re.sub(r"'\s*,\s*'[^']+'\s*:.*$", "", cleaned, flags=re.DOTALL)
    return cleaned.replace("\\n", "\n").strip()


def clean_financebench_data():
    if not QA_FILE.exists():
        print(f"ERROR: {QA_FILE} not found")
        return False

    # Read original records
    records = [json.loads(line) for line in open(QA_FILE, encoding="utf-8")]
    print(f"Read {len(records)} records from {QA_FILE.name}")

    # Check corruption level
    corrupted = sum(1 for r in records if r.get("context_text", "").strip().startswith("{"))
    print(f"Found {corrupted}/{len(records)} records with corrupted context_text")

    if corrupted == 0:
        print("No corruption detected, nothing to fix.")
        return True

    # Backup original
    print(f"Backing up to {BACKUP_FILE.name}...")
    with open(BACKUP_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Fix each record
    fixed_records = []
    for r in records:
        original_context = r.get("context_text", "")

        if original_context.strip().startswith("{"):
            # Parse and clean
            cleaned = parse_corrupted_context(original_context)
            r["context_text"] = cleaned
            r["doc_length_chars"] = len(cleaned)
            # Recalculate structure score
            r["doc_structure_score"] = structure_score(cleaned)

        fixed_records.append(r)

    # Write fixed data
    print(f"Writing fixed data to {QA_FILE.name}...")
    with open(QA_FILE, "w", encoding="utf-8") as f:
        for r in fixed_records:
            f.write(json.dumps(r) + "\n")

    # Summary
    avg_len_before = sum(len(json.loads(line).get("context_text", ""))
                         for line in open(BACKUP_FILE)) / len(records)
    avg_len_after = sum(r["doc_length_chars"] for r in fixed_records) / len(fixed_records)

    print(f"\nFix complete:")
    print(f"  Records fixed: {corrupted}")
    print(f"  Avg context_text length: {avg_len_before:.0f} -> {avg_len_after:.0f} chars")

    # Show sample
    print("\nSample cleaned record:")
    sample = fixed_records[0]
    print(f"  ID: {sample['id']}")
    print(f"  Question: {sample['question'][:80]}...")
    print(f"  Context (first 300 chars): {sample['context_text'][:300]}...")

    return True


def structure_score(text: str) -> float:
    """
    0–1 score for how structurally rich a document is.
    """
    if not text:
        return 0.0
    lines = text.splitlines()
    n = max(len(lines), 1)
    headers = sum(1 for l in lines if l.strip().startswith("#"))
    bullets = sum(1 for l in lines if l.strip().startswith(("-", "*", "•")))
    numbered = sum(1 for l in lines if len(l.strip()) > 2 and l.strip()[0].isdigit() and l.strip()[1] in ".)")
    return round(min((headers * 3 + bullets + numbered) / n, 1.0), 4)


if __name__ == "__main__":
    print("=== Fixing FinanceBench Data Corruption ===\n")
    success = clean_financebench_data()
    if success:
        print("\nDone. You should now re-run the following to regenerate indexes and answers:")
        print("  python scripts/02_build_indexes.py")
        print("  python scripts/03_build_pageindex_trees.py")
        print("  python scripts/04_run_rag_eval.py")
        print("  python scripts/05_run_pageindex_eval.py")
        print("  python scripts/06_llm_judge.py")
