"""
Script 04: Run Dense RAG and BM25 evaluation on all 600 QA pairs.

Both use the same LLM (gpt-4o-mini via Copilot) for answer generation.
The ONLY difference is how context is retrieved.

IEEE note: Running both baselines in a single script ensures identical
LLM calls/prompts for fair comparison. Same temperature=0, same max_tokens.
"""

import json, pickle, time, re, sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from setup_copilot import get_client  # noqa

ROOT    = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")
DATA    = ROOT / "data"
RIDX    = ROOT / "results" / "rag_index"
BIDX    = ROOT / "results" / "bm25_index"
RAG_OUT = ROOT / "results" / "rag_answers"
BM25_OUT= ROOT / "results" / "bm25_answers"
RAG_OUT.mkdir(parents=True, exist_ok=True)
BM25_OUT.mkdir(parents=True, exist_ok=True)

DOMAINS   = ["financebench", "cuad", "qasper", "techqa"]
MODEL     = "gpt-4o-mini"
MAX_TOK   = 256
TOP_K     = 5

PROMPT = """\
Answer the question using ONLY the context below. Be concise and precise.
If the answer is not in the context, reply exactly: NOT FOUND

Context:
{context}

Question: {question}

Answer:"""


# ── Metrics ──────────────────────────────────────────────────────────────────

def normalize(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r'\b(a|an|the)\b', ' ', t)
    t = re.sub(r'[^\w\s]', ' ', t)
    return " ".join(t.split())

def em(pred, gold): return int(normalize(pred) == normalize(gold))

def f1(pred, gold):
    p_t = set(normalize(pred).split())
    g_t = set(normalize(gold).split())
    if not p_t or not g_t: return 0.0
    c = p_t & g_t
    if not c: return 0.0
    pr, rc = len(c)/len(p_t), len(c)/len(g_t)
    return round(2*pr*rc/(pr+rc), 4)

def answer_and_score(client, context, question, gold):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content":
                   PROMPT.format(context=context[:3500], question=question)}],
        max_tokens=MAX_TOK, temperature=0,
    )
    pred    = resp.choices[0].message.content.strip()
    tok_in  = resp.usage.prompt_tokens
    tok_out = resp.usage.completion_tokens
    return pred, em(pred, gold), f1(pred, gold), tok_in, tok_out


# ── Dense RAG ─────────────────────────────────────────────────────────────────

def run_rag(domain: str, client):
    out_path = RAG_OUT / f"{domain}.jsonl"
    done_ids = {json.loads(l)["id"] for l in open(out_path)} if out_path.exists() else set()
    records  = [json.loads(l) for l in open(DATA / domain / "qa_pairs.jsonl")]
    todo     = [r for r in records if r["id"] not in done_ids]
    if not todo:
        print(f"    RAG  [{domain}]: already complete")
        return

    from langchain_community.embeddings import HuggingFaceEmbeddings
    with open(RIDX / f"{domain}.pkl", "rb") as f:
        db = pickle.load(f)
    embed = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",
                                   model_kwargs={"device": "cpu"},
                                   encode_kwargs={"normalize_embeddings": True})
    db.embedding_function = embed.embed_query

    for r in todo:
        t0   = time.time()
        docs = db.similarity_search(r["question"], k=TOP_K)
        ctx  = "\n---\n".join(d.page_content for d in docs)
        pred, em_, f1_, tok_in, tok_out = answer_and_score(
            client, ctx, r["question"], r["answer"])
        with open(out_path, "a") as f_out:
            f_out.write(json.dumps({
                "id": r["id"], "domain": domain,
                "question": r["question"], "gold_answer": r["answer"],
                "pred_answer": pred, "em": em_, "f1": f1_,
                "tokens_in": tok_in, "tokens_out": tok_out,
                "latency_s": round(time.time()-t0, 3),
                "method": "rag",
                "doc_length_chars": r.get("doc_length_chars", 0),
                "doc_structure_score": r.get("doc_structure_score", 0.0),
            }) + "\n")

    rows   = [json.loads(l) for l in open(out_path)]
    print(f"    RAG  [{domain}]: EM={sum(r['em'] for r in rows)/len(rows):.3f}  "
          f"F1={sum(r['f1'] for r in rows)/len(rows):.3f}  (n={len(rows)})")


# ── BM25 ─────────────────────────────────────────────────────────────────────

def run_bm25(domain: str, client):
    out_path = BM25_OUT / f"{domain}.jsonl"
    done_ids = {json.loads(l)["id"] for l in open(out_path)} if out_path.exists() else set()
    records  = [json.loads(l) for l in open(DATA / domain / "qa_pairs.jsonl")]
    todo     = [r for r in records if r["id"] not in done_ids]
    if not todo:
        print(f"    BM25 [{domain}]: already complete")
        return

    with open(BIDX / f"{domain}.pkl", "rb") as f:
        idx_data = pickle.load(f)
    bm25   = idx_data["bm25"]
    chunks = idx_data["chunks"]

    for r in todo:
        t0     = time.time()
        tokens = r["question"].lower().split()
        scores = bm25.get_scores(tokens)
        # Get top-k indices
        import numpy as np
        top_k  = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:TOP_K]
        ctx    = "\n---\n".join(chunks[i] for i in top_k)
        pred, em_, f1_, tok_in, tok_out = answer_and_score(
            client, ctx, r["question"], r["answer"])
        with open(out_path, "a") as f_out:
            f_out.write(json.dumps({
                "id": r["id"], "domain": domain,
                "question": r["question"], "gold_answer": r["answer"],
                "pred_answer": pred, "em": em_, "f1": f1_,
                "tokens_in": tok_in, "tokens_out": tok_out,
                "latency_s": round(time.time()-t0, 3),
                "method": "bm25",
                "doc_length_chars": r.get("doc_length_chars", 0),
                "doc_structure_score": r.get("doc_structure_score", 0.0),
            }) + "\n")

    rows   = [json.loads(l) for l in open(out_path)]
    print(f"    BM25 [{domain}]: EM={sum(r['em'] for r in rows)/len(rows):.3f}  "
          f"F1={sum(r['f1'] for r in rows)/len(rows):.3f}  (n={len(rows)})")


if __name__ == "__main__":
    print("=== Step 4: RAG + BM25 Evaluation ===\n")
    client = get_client()
    for domain in DOMAINS:
        print(f"  Domain: {domain}")
        run_rag(domain, client)
        run_bm25(domain, client)
    print("\nStep 4 complete.")
