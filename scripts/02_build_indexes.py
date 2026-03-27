"""
Script 02: Build retrieval indexes for ALL THREE systems.

  System A — Dense RAG:  FAISS + sentence-transformers (all-MiniLM-L6-v2)
  System B — BM25:       rank-bm25 (sparse lexical retrieval, classic IR baseline)
  System C — PageIndex:  Hierarchical LLM tree built via PageIndex repo

IEEE rationale for 3 systems:
  - BM25 is required as a classical baseline. Reviewers will ask for it.
  - Dense RAG is the neural baseline PageIndex claims to beat.
  - PageIndex is the proposed method. Comparing all three answers:
    "Does PageIndex beat both kinds of retrieval, or just dense?"

BM25 and FAISS are both built here. PageIndex trees are built in script 03
because they require LLM calls (Copilot token).
"""

import json, pickle
from pathlib import Path

ROOT   = Path(__file__).parent.parent
DATA   = ROOT / "data"
RIDX   = ROOT / "results" / "rag_index"
BIDX   = ROOT / "results" / "bm25_index"
RIDX.mkdir(parents=True, exist_ok=True)
BIDX.mkdir(parents=True, exist_ok=True)

DOMAINS     = ["financebench", "cuad", "qasper", "techqa"]
EMBED_MODEL = "all-MiniLM-L6-v2"


def get_doc_text(r: dict) -> str:
    for k in ["context_text", "evidence"]:
        v = str(r.get(k, "")).strip()
        if len(v) > 80:
            return v
    return f"{r.get('question', '')} {r.get('answer', '')}"


# ── Dense RAG index ──────────────────────────────────────────────────────────

def build_rag_index(domain: str):
    out = RIDX / f"{domain}.pkl"
    if out.exists():
        print(f"    Dense FAISS [{domain}]: already built")
        return
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    records  = [json.loads(l) for l in open(DATA / domain / "qa_pairs.jsonl")]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks, metas = [], []
    for r in records:
        for chunk in splitter.split_text(get_doc_text(r)):
            chunks.append(chunk)
            metas.append({"id": r["id"], "domain": domain})

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    db = FAISS.from_texts(chunks, embeddings, metadatas=metas)
    with open(out, "wb") as f:
        pickle.dump(db, f)
    print(f"    Dense FAISS [{domain}]: {len(chunks)} chunks indexed")


# ── BM25 index ───────────────────────────────────────────────────────────────

def build_bm25_index(domain: str):
    out = BIDX / f"{domain}.pkl"
    if out.exists():
        print(f"    BM25        [{domain}]: already built")
        return
    from rank_bm25 import BM25Okapi
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    records  = [json.loads(l) for l in open(DATA / domain / "qa_pairs.jsonl")]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks, metas = [], []
    for r in records:
        for chunk in splitter.split_text(get_doc_text(r)):
            chunks.append(chunk)
            metas.append({"id": r["id"], "domain": domain})

    tokenized = [c.lower().split() for c in chunks]
    bm25      = BM25Okapi(tokenized)

    with open(out, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks, "metas": metas}, f)
    print(f"    BM25        [{domain}]: {len(chunks)} chunks indexed")


if __name__ == "__main__":
    print("=== Step 2: Building Retrieval Indexes ===\n")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install",
        "langchain", "langchain-community", "sentence-transformers",
        "faiss-cpu", "rank-bm25", "-q"])
    print(f"  Embedding model: {EMBED_MODEL} (local, no API)\n")

    for domain in DOMAINS:
        build_rag_index(domain)
        build_bm25_index(domain)

    print("\nStep 2 complete.")
