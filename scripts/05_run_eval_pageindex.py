"""
Script 05: Run PageIndex evaluation on all 600 QA pairs.

Two-step retrieval per question (replicating PageIndex's algorithm):
  Step A — Tree traversal: LLM reads tree summary and identifies relevant node(s)
  Step B — Answer generation: LLM answers from the identified node content

Same LLM (gpt-4o-mini), same answer prompt as RAG/BM25 — only retrieval differs.
"""

import json, re, time, sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from setup_copilot import get_client  # noqa

ROOT    = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")
DATA    = ROOT / "data"
TREES   = ROOT / "results" / "pageindex_trees"
OUT_DIR = ROOT / "results" / "pageindex_answers"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOMAINS   = ["financebench", "cuad", "qasper", "techqa"]
MODEL     = "gpt-4o-mini"   # same as RAG/BM25
MAX_TOK   = 256

TREE_SEARCH_PROMPT = """\
You are navigating a hierarchical document index to answer a question.

Each entry below is a node in the index tree (node_id: summary/content).
Identify the most relevant node(s) for answering the question.
State the node_id(s) on the first line, then provide your answer.

Index:
{tree_summary}

Question: {question}

Most relevant node_id(s) and answer:"""

ANSWER_PROMPT = """\
Answer the question using ONLY the section content below. Be concise and precise.
If the answer is not in the content, reply exactly: NOT FOUND

Section content:
{content}

Question: {question}

Answer:"""


# ── Metrics ──────────────────────────────────────────────────────────────────

def normalize(t):
    t = t.lower().strip()
    t = re.sub(r'\b(a|an|the)\b', ' ', t)
    t = re.sub(r'[^\w\s]', ' ', t)
    return " ".join(t.split())

def em(pred, gold):  return int(normalize(pred) == normalize(gold))
def f1(pred, gold):
    p_t = set(normalize(pred).split())
    g_t = set(normalize(gold).split())
    if not p_t or not g_t: return 0.0
    c = p_t & g_t
    if not c: return 0.0
    pr, rc = len(c)/len(p_t), len(c)/len(g_t)
    return round(2*pr*rc/(pr+rc), 4)


# ── Tree helpers ──────────────────────────────────────────────────────────────

def make_tree_summary(nodes: list, max_nodes: int = 30) -> str:
    """
    Produce a structured index summary for the tree-search LLM prompt.
    Prioritize higher-level nodes (summaries) over leaves.
    """
    # Sort: highest level (root/parent) first so the LLM sees coarse structure first
    sorted_nodes = sorted(nodes, key=lambda n: -n.get("level", 0))
    lines = []
    for node in sorted_nodes[:max_nodes]:
        nid     = node.get("node_id", "?")
        level   = node.get("level", 0)
        # Use summary if available (parent nodes), else truncate content
        text = node.get("summary") or node.get("content", "")
        text = text[:200].replace("\n", " ")
        indent = "  " * (2 - min(level, 2))
        lines.append(f"{indent}[{nid}] (L{level}): {text}")
    return "\n".join(lines)


def best_node_content(nodes: list, search_output: str, question: str) -> str:
    """
    Extract the content of the node the LLM identified in Step A.
    Falls back to leaf-level search by question keywords if no node_id found.
    """
    # Parse node_ids from the first line of search_output
    first_line = search_output.split("\n")[0] if search_output else ""
    for node in nodes:
        nid = str(node.get("node_id", ""))
        if nid and nid in first_line:
            # Found referenced node — return its content + parent content
            content = node.get("content", "")
            # Also grab any children content for completeness
            return content[:3000]

    # Fallback: find leaf node whose content best matches question keywords
    q_words = set(question.lower().split())
    scored  = []
    for node in nodes:
        if node.get("level", 0) != 0:  # leaves only
            continue
        content = node.get("content", "").lower()
        score   = sum(1 for w in q_words if w in content and len(w) > 3)
        scored.append((score, node))
    if scored:
        best = max(scored, key=lambda x: x[0])[1]
        return best.get("content", "")[:3000]

    # Last resort: concatenate first 3 leaf nodes
    leaves = [n for n in nodes if n.get("level", 0) == 0][:3]
    return " ".join(n.get("content", "") for n in leaves)[:3000]


def get_domain_tree(domain: str) -> dict:
    tree_path = TREES / f"{domain}_tree.json"
    if tree_path.exists():
        return json.load(open(tree_path))
    return {"nodes": [], "method": "missing"}


# ── Main ──────────────────────────────────────────────────────────────────────

def run_pageindex(domain: str, client):
    out_path = OUT_DIR / f"{domain}.jsonl"
    done_ids = {json.loads(l)["id"] for l in open(out_path)} if out_path.exists() else set()
    records  = [json.loads(l) for l in open(DATA / domain / "qa_pairs.jsonl")]
    todo     = [r for r in records if r["id"] not in done_ids]
    if not todo:
        print(f"    PI   [{domain}]: already complete")
        return

    tree    = get_domain_tree(domain)
    nodes   = tree.get("nodes", [])
    method  = tree.get("method", "unknown")
    summary = make_tree_summary(nodes)

    if not nodes:
        print(f"    PI   [{domain}]: WARNING — no tree nodes, check script 03")

    for r in todo:
        t0 = time.time()

        # Step A: tree traversal
        resp_a = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TREE_SEARCH_PROMPT.format(
                tree_summary=summary[:3000],
                question=r["question"],
            )}],
            max_tokens=250, temperature=0,
        )
        search_out = resp_a.choices[0].message.content.strip()
        tok_a_in   = resp_a.usage.prompt_tokens
        tok_a_out  = resp_a.usage.completion_tokens

        # Step B: answer from identified node
        content = best_node_content(nodes, search_out, r["question"])
        resp_b  = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": ANSWER_PROMPT.format(
                content=content, question=r["question"],
            )}],
            max_tokens=MAX_TOK, temperature=0,
        )
        pred     = resp_b.choices[0].message.content.strip()
        latency  = round(time.time()-t0, 3)
        tok_b_in = resp_b.usage.prompt_tokens
        tok_b_out= resp_b.usage.completion_tokens

        with open(out_path, "a") as f_out:
            f_out.write(json.dumps({
                "id": r["id"], "domain": domain,
                "question": r["question"], "gold_answer": r["answer"],
                "pred_answer": pred,
                "tree_search_out": search_out[:200],
                "em": em(pred, r["answer"]),
                "f1": f1(pred, r["answer"]),
                "tokens_in":   tok_a_in + tok_b_in,
                "tokens_out":  tok_a_out + tok_b_out,
                "latency_s":   latency,
                "method": "pageindex",
                "tree_method": method,
                "doc_length_chars":    r.get("doc_length_chars", 0),
                "doc_structure_score": r.get("doc_structure_score", 0.0),
            }) + "\n")

    rows = [json.loads(l) for l in open(out_path)]
    print(f"    PI   [{domain}]: EM={sum(r['em'] for r in rows)/len(rows):.3f}  "
          f"F1={sum(r['f1'] for r in rows)/len(rows):.3f}  (n={len(rows)}  "
          f"tree={method})")


if __name__ == "__main__":
    print("=== Step 5: PageIndex Evaluation ===\n")
    client = get_client()
    for domain in DOMAINS:
        print(f"  Domain: {domain}")
        run_pageindex(domain, client)
    print("\nStep 5 complete.")
