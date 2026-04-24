"""
Run RAG evaluation using different models per domain to work around
per-model daily rate limits (150 req/day per model on GitHub Models API).

Each model is a separate rate limit bucket, so we can parallelize.
Within each domain, both RAG and PageIndex use the SAME model for fair comparison.
"""
import json, pickle, time, re, sys, os
from pathlib import Path
from dotenv import load_dotenv
import openai

sys.path.insert(0, str(Path(__file__).parent))
from setup_copilot import get_client

BASE = Path(__file__).parent.parent
load_dotenv(BASE / ".env")
DATA = BASE / "data"
INDEX = BASE / "results" / "rag_index"
OUT_DIR = BASE / "results" / "rag_answers"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Different model per domain to spread across rate limit buckets.
# Within each domain, both RAG and PageIndex use the SAME model.
DOMAIN_MODELS = {
    "financebench": "gpt-4.1-mini",
    "cuad": "gpt-4.1-nano",
    "qasper": "Phi-4",
    "techqa": "Cohere-command-r-08-2024",
}

MAX_TOKENS = 256

PROMPT = """\
Answer the question using only the context below. Be concise.
If the answer is not in the context, reply "NOT FOUND".

Context:
{context}

Question: {question}

Answer:"""


def normalize(text):
    text = text.lower().strip()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return " ".join(text.split())

def exact_match(pred, gold):
    return int(normalize(pred) == normalize(gold))

def f1(pred, gold):
    p_t = set(normalize(pred).split())
    g_t = set(normalize(gold).split())
    if not p_t or not g_t: return 0.0
    common = p_t & g_t
    if not common: return 0.0
    p, r = len(common)/len(p_t), len(common)/len(g_t)
    return 2*p*r/(p+r)


def run_domain(domain, client):
    model = DOMAIN_MODELS[domain]
    out_path = OUT_DIR / f"{domain}.jsonl"
    done_ids = set()
    if out_path.exists():
        done_ids = {json.loads(l)["id"] for l in open(out_path)}

    records = [json.loads(l) for l in open(DATA / domain / "qa_pairs.jsonl")]
    remaining = [r for r in records if r["id"] not in done_ids]

    if not remaining:
        print(f"  {domain}: already complete ({len(records)}/{len(records)})")
        return

    with open(INDEX / f"{domain}.pkl", "rb") as f:
        db = pickle.load(f)

    from langchain_community.embeddings import HuggingFaceEmbeddings
    embed = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    db.embedding_function = embed.embed_query

    print(f"  {domain}: model={model}, {len(done_ids)} done, {len(remaining)} remaining...",
          flush=True)

    for i, r in enumerate(remaining):
        time.sleep(4.5)  # Rate pacing
        t0 = time.time()
        docs = db.similarity_search(r["question"], k=5)
        context = "\n---\n".join(d.page_content for d in docs)[:3500]

        for attempt in range(5):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content":
                                PROMPT.format(context=context, question=r["question"])}],
                    max_tokens=MAX_TOKENS, temperature=0,
                )
                break
            except openai.RateLimitError:
                wait = min(60 * (attempt + 1), 120)
                print(f"    Rate limited on {model}, waiting {wait}s...", flush=True)
                time.sleep(wait)
        else:
            print(f"    SKIP {r['id']}: rate limit exhausted", flush=True)
            continue

        latency = round(time.time() - t0, 3)
        pred = resp.choices[0].message.content.strip()

        result = {
            "id": r["id"], "domain": domain,
            "question": r["question"], "gold_answer": r["answer"],
            "pred_answer": pred,
            "em": exact_match(pred, r["answer"]),
            "f1": round(f1(pred, r["answer"]), 4),
            "tokens_in": resp.usage.prompt_tokens,
            "tokens_out": resp.usage.completion_tokens,
            "latency_s": latency, "method": "rag", "model": model,
        }
        with open(out_path, "a") as f_out:
            f_out.write(json.dumps(result) + "\n")

        done_now = len(done_ids) + i + 1
        if done_now % 20 == 0:
            print(f"    [{done_now}/{len(records)}] F1={result['f1']:.2f}", flush=True)

    all_rows = [json.loads(l) for l in open(out_path)]
    nf = sum(1 for r in all_rows if "NOT FOUND" in r["pred_answer"].upper())
    f1_avg = sum(r["f1"] for r in all_rows) / len(all_rows)
    print(f"  {domain}: DONE n={len(all_rows)} NF={nf} ({100*nf/len(all_rows):.1f}%) "
          f"F1={f1_avg:.3f}", flush=True)


if __name__ == "__main__":
    print("=== RAG Evaluation (multi-model) ===\n")
    client = get_client()
    for domain in ["financebench", "cuad", "qasper", "techqa"]:
        run_domain(domain, client)
    print("\nDone.")
