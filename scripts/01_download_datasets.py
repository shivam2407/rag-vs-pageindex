"""
Script 01: Download 600 QA pairs (150 per domain) from HuggingFace.

IEEE-bar additions vs naive version:
  - Random seed fixed + logged for reproducibility
  - Document length (chars) and structure score recorded per record
    → used later for RQ2 characteristic analysis
  - Version metadata saved alongside each dataset split
  - Stratified sampling (ensures length distribution is not skewed)
"""

import json, random, hashlib
from pathlib import Path
from datetime import datetime

ROOT   = Path(__file__).parent.parent
DATA   = ROOT / "data"
SEED   = 42
N      = 150           # QA pairs per domain
random.seed(SEED)


# ── Shared helpers ─────────────────────────────────────────────────────────

def structure_score(text: str) -> float:
    """
    0–1 score for how structurally rich a document is.
    Counts markdown headers, lists, numbered sections as signals.
    Used in RQ2 to test whether PageIndex advantage correlates with structure.
    """
    if not text:
        return 0.0
    lines = text.splitlines()
    n     = max(len(lines), 1)
    headers  = sum(1 for l in lines if l.strip().startswith("#"))
    bullets  = sum(1 for l in lines if l.strip().startswith(("-", "*", "•")))
    numbered = sum(1 for l in lines if len(l.strip()) > 2 and l.strip()[0].isdigit() and l.strip()[1] in ".)")
    return round(min((headers * 3 + bullets + numbered) / n, 1.0), 4)


def enrich(record: dict) -> dict:
    """Add doc_length_chars and doc_structure_score to every record."""
    text = record.get("context_text", record.get("evidence", ""))
    record["doc_length_chars"]    = len(text)
    record["doc_structure_score"] = structure_score(text)
    return record


def save_jsonl(path: Path, records: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"    Saved {len(records)} records  (avg_len="
          f"{sum(r['doc_length_chars'] for r in records)//len(records)} chars)")


def save_meta(domain: str, ds_info: dict):
    meta = {
        "domain":      domain,
        "n":           N,
        "seed":        SEED,
        "downloaded":  datetime.utcnow().isoformat() + "Z",
        "source":      ds_info,
    }
    (DATA / domain / "meta.json").write_text(json.dumps(meta, indent=2))


def stratified_sample(records: list, n: int, key: str = "doc_length_chars") -> list:
    """
    Stratified sample by document length to avoid length-bias in results.
    Splits into 3 tertiles and samples proportionally.
    """
    if len(records) <= n:
        return records
    sorted_r  = sorted(records, key=lambda r: r.get(key, 0))
    tertile   = len(sorted_r) // 3
    per_bin   = n // 3
    sample    = []
    for i in range(3):
        start = i * tertile
        end   = (i + 1) * tertile if i < 2 else len(sorted_r)
        chunk = sorted_r[start:end]
        sample.extend(random.sample(chunk, min(per_bin, len(chunk))))
    # Fill remainder randomly from leftover
    remaining = [r for r in records if r not in sample]
    sample   += random.sample(remaining, max(0, n - len(sample)))
    return sample[:n]


# ── DOMAIN 1: FinanceBench ──────────────────────────────────────────────────

def download_financebench():
    out = DATA / "financebench" / "qa_pairs.jsonl"
    if out.exists():
        print("  FinanceBench: already downloaded, skipping")
        return
    print("  Downloading FinanceBench...")
    from datasets import load_dataset
    ds = load_dataset("PatronusAI/financebench", split="train")
    records = []
    for row in ds:
        evidence = row.get("evidence", "")
        if isinstance(evidence, list):
            evidence = "\n".join(str(e) for e in evidence)
        r = enrich({
            "id":           row.get("question_id", f"fb_{len(records)}"),
            "domain":       "finance",
            "question":     row["question"],
            "answer":       str(row["answer"]).strip(),
            "context_doc":  row.get("doc_name", ""),
            "context_text": str(evidence),
        })
        if r["answer"]:
            records.append(r)
    sample = stratified_sample(records, N)
    save_jsonl(out, sample)
    save_meta("financebench", {"hf_id": "PatronusAI/financebench", "split": "train"})


# ── DOMAIN 2: CUAD (Legal Contracts) ───────────────────────────────────────

def download_cuad():
    out = DATA / "cuad" / "qa_pairs.jsonl"
    if out.exists():
        print("  CUAD: already downloaded, skipping")
        return
    print("  Downloading CUAD (SQuAD-format JSON from HF repo)...")
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("theatticusproject/cuad", filename="CUAD_v1/CUAD_v1.json",
                           repo_type="dataset")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    records = []
    for article in data.get("data", []):
        title = article.get("title", "")
        for para in article.get("paragraphs", []):
            context = para.get("context", "")
            for qa in para.get("qas", []):
                if qa.get("is_impossible", False):
                    continue
                answers = qa.get("answers", [])
                if not answers:
                    continue
                answer_text = answers[0].get("text", "").strip()
                if len(answer_text) < 10:
                    continue
                r = enrich({
                    "id":           qa.get("id", f"cuad_{len(records)}"),
                    "domain":       "legal",
                    "question":     qa.get("question", ""),
                    "answer":       answer_text,
                    "context_doc":  title,
                    "context_text": context[:5000],
                })
                records.append(r)
    print(f"    Found {len(records)} answerable QA pairs from CUAD")
    sample = stratified_sample(records, N)
    save_jsonl(out, sample)
    save_meta("cuad", {"source": "theatticusproject/cuad", "file": "CUAD_v1/CUAD_v1.json"})


# ── DOMAIN 3: QASPER (Scientific Papers) ───────────────────────────────────

def download_qasper():
    out = DATA / "qasper" / "qa_pairs.jsonl"
    if out.exists():
        print("  QASPER: already downloaded, skipping")
        return
    print("  Downloading QASPER (from S3 archive)...")
    import urllib.request, tarfile, tempfile
    url = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
    tmp = Path(tempfile.mkdtemp())
    tgz = tmp / "qasper.tgz"
    urllib.request.urlretrieve(url, tgz)
    with tarfile.open(tgz, "r:gz") as tar:
        tar.extractall(tmp, filter="data")
    train_file = None
    for f in tmp.rglob("*train*.json"):
        train_file = f
        break
    if not train_file:
        raise FileNotFoundError(f"No train JSON found in {list(tmp.rglob('*.json'))}")
    data = json.loads(train_file.read_text(encoding="utf-8"))
    records = []
    for paper_id, paper in data.items():
        full_text = paper.get("full_text", [])
        sections = []
        if isinstance(full_text, list):
            for section in full_text:
                name = section.get("section_name", "")
                paras = section.get("paragraphs", [])
                if isinstance(paras, list):
                    sections.append(f"## {name}\n" + " ".join(str(p) for p in paras))
        elif isinstance(full_text, dict):
            for name, paras in zip(
                full_text.get("section_name", []),
                full_text.get("paragraphs", [])
            ):
                if isinstance(paras, list):
                    sections.append(f"## {name}\n" + " ".join(str(p) for p in paras))
        doc_text = "\n\n".join(sections)[:8000]
        for i, qas in enumerate(paper.get("qas", [])):
            question = qas.get("question", "")
            answer = ""
            for ans in qas.get("answers", []):
                a = ans.get("answer", {})
                ffa = a.get("free_form_answer", "") if isinstance(a, dict) else ""
                if ffa:
                    answer = ffa
                    break
            if not question or not answer or len(answer) < 5:
                continue
            r = enrich({
                "id":           f"{paper_id}_q{i}",
                "domain":       "science",
                "question":     question,
                "answer":       answer.strip(),
                "context_doc":  paper.get("title", ""),
                "context_text": doc_text,
            })
            records.append(r)
        if len(records) >= 600:
            break
    print(f"    Found {len(records)} QA pairs from QASPER")
    sample = stratified_sample(records, N)
    save_jsonl(out, sample)
    save_meta("qasper", {"source": "qasper-dataset.s3.us-west-2.amazonaws.com", "version": "v0.3"})


# ── DOMAIN 4: TechQA ───────────────────────────────────────────────────────

def download_techqa():
    out = DATA / "techqa" / "qa_pairs.jsonl"
    if out.exists():
        print("  TechQA: already downloaded, skipping")
        return
    print("  Downloading TechQA (SQuAD tech-filtered fallback)...")
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad_v2", split="train")
    tech_kw = {"software", "computer", "programming", "internet", "server",
               "database", "algorithm", "network", "linux", "windows",
               "python", "java", "api", "cloud", "technology", "digital",
               "processor", "memory", "hardware", "encryption", "http",
               "compiler", "kernel", "protocol", "bandwidth", "binary"}
    records = []
    for row in ds:
        answers = row.get("answers", {})
        texts = answers.get("text", []) if isinstance(answers, dict) else []
        if not texts or len(texts[0]) < 5:
            continue
        context = row.get("context", "")
        question = row.get("question", "")
        combined = (context + " " + question).lower()
        if not any(kw in combined for kw in tech_kw):
            continue
        r = enrich({
            "id":           row.get("id", f"techqa_{len(records)}"),
            "domain":       "technology",
            "question":     question,
            "answer":       texts[0].strip(),
            "context_doc":  row.get("title", ""),
            "context_text": context[:5000],
        })
        records.append(r)
    print(f"    Found {len(records)} tech-related QA pairs")
    sample = stratified_sample(records, N)
    save_jsonl(out, sample)
    save_meta("techqa", {"source": "rajpurkar/squad_v2", "note": "tech-keyword filtered"})


if __name__ == "__main__":
    print("=== Step 1: Downloading Datasets ===\n")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "datasets", "huggingface_hub", "-q"])

    download_financebench()
    download_cuad()
    download_qasper()
    download_techqa()

    print("\nDataset summary:")
    total = 0
    for domain in ["financebench", "cuad", "qasper", "techqa"]:
        p = DATA / domain / "qa_pairs.jsonl"
        if not p.exists():
            print(f"  ✗ {domain}: MISSING")
            continue
        rows = [json.loads(l) for l in open(p)]
        total += len(rows)
        avg_len = sum(r["doc_length_chars"] for r in rows) // max(len(rows), 1)
        avg_str = sum(r["doc_structure_score"] for r in rows) / max(len(rows), 1)
        print(f"  ✓ {domain:15s}: n={len(rows)}  "
              f"avg_doc_len={avg_len:6d}  avg_struct={avg_str:.3f}")
    print(f"\n  Total: {total} / {N*4} target\n")
    print(f"  Seed: {SEED}  (fixed for reproducibility)")
    print("Step 1 complete.")
