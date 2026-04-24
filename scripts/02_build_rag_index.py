"""
Script 02: Build FAISS vector index for RAG baseline.
Uses sentence-transformers (all-MiniLM-L6-v2) — free, local, no API needed.
~80MB one-time model download. Idempotent.
"""

import json, pickle
from pathlib import Path

BASE  = Path(__file__).parent.parent
DATA  = BASE / "data"
INDEX = BASE / "results" / "rag_index"
INDEX.mkdir(parents=True, exist_ok=True)

DOMAINS    = ["financebench", "cuad", "qasper", "techqa"]
EMBED_MODEL = "all-MiniLM-L6-v2"   # 80MB, 384-dim, fast & good


def get_doc_text(record: dict) -> str:
    """Best available document text for a QA record."""
    for key in ["context_text", "evidence"]:
        val = str(record.get(key, "")).strip()
        if len(val) > 80:
            return val
    # Fallback: concatenate question + answer as minimal context
    return f"Q: {record.get('question', '')} A: {record.get('answer', '')}"


def build_index(domain: str):
    out_path = INDEX / f"{domain}.pkl"
    if out_path.exists():
        print(f"  {domain}: index already exists, skipping")
        return

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    qa_path = DATA / domain / "qa_pairs.jsonl"
    records = [json.loads(l) for l in open(qa_path)]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks, metas = [], []
    for r in records:
        text = get_doc_text(r)
        for chunk in splitter.split_text(text):
            chunks.append(chunk)
            metas.append({"id": r["id"], "domain": domain})

    print(f"  {domain}: {len(records)} records → {len(chunks)} chunks, embedding...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    db = FAISS.from_texts(chunks, embeddings, metadatas=metas)
    with open(out_path, "wb") as f:
        pickle.dump(db, f)
    print(f"  {domain}: index saved ({len(chunks)} chunks)")


if __name__ == "__main__":
    print("=== Step 2: Building RAG Vector Indexes ===\n")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install",
        "langchain", "langchain-community", "sentence-transformers",
        "faiss-cpu", "-q"])
    print(f"  Embedding model: {EMBED_MODEL} (local, no API)\n")

    for domain in DOMAINS:
        build_index(domain)

    print("\nStep 2 complete.")
