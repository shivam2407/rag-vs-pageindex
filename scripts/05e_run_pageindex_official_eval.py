"""
Script 05e: PageIndex evaluation using VectifyAI's retrieval algorithm.

This script reimplements tree-based retrieval as described in the
PageIndex paper:
  1. Load pre-built tree (from 03b) for the question's document
  2. Show LLM the tree structure (titles + hierarchy, NO text)
  3. LLM selects which nodes to read (top-down navigation)
  4. Retrieve text content for selected nodes
  5. LLM answers from retrieved content

This mirrors the PageIndex retrieval pattern:
  get_document_structure() -> LLM navigates -> get_page_content()

Same generation model per domain as RAG for controlled comparison.
Idempotent - resumes from where it left off.
"""

import json, re, time, sys, os, hashlib
from pathlib import Path
from dotenv import load_dotenv
import openai

sys.path.insert(0, str(Path(__file__).parent))
from setup_copilot import get_client

ROOT    = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")
DATA    = ROOT / "data"
TREES   = ROOT / "results" / "pageindex_trees"
OUT_DIR = ROOT / "results" / "pageindex_answers"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VENDOR  = ROOT / "vendor" / "PageIndex"
sys.path.insert(0, str(VENDOR))

DOMAINS   = ["financebench", "cuad", "qasper", "techqa"]
MAX_TOKENS = 256

# MUST match the per-domain models in 04b_run_rag_multimodel.py
# so that within each domain, both RAG and PageIndex use the same LLM.
DOMAIN_MODELS = {
    "financebench": "gpt-4.1-mini",
    "cuad": "gpt-4.1-nano",
    "qasper": "Phi-4",
    "techqa": "Cohere-command-r-08-2024",
}


def llm_call(client, **kwargs):
    """LLM call with rate-limit retry."""
    for attempt in range(5):
        try:
            return client.chat.completions.create(**kwargs)
        except openai.RateLimitError:
            wait = min(60 * (attempt + 1), 120)
            print(f"    Rate limited, waiting {wait}s...", flush=True)
            time.sleep(wait)
    raise RuntimeError("Rate limit exhausted after 5 retries")


# -- Metrics ----------------------------------------------------------------

def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return " ".join(text.split())

def exact_match(pred, gold):
    return int(normalize(pred) == normalize(gold))

def f1(pred, gold):
    p_t = set(normalize(pred).split())
    g_t = set(normalize(gold).split())
    if not p_t or not g_t:
        return 0.0
    common = p_t & g_t
    if not common:
        return 0.0
    p, r = len(common) / len(p_t), len(common) / len(g_t)
    return 2 * p * r / (p + r)


# -- Tree helpers using VectifyAI's retrieve module -------------------------

def format_tree_for_navigation(structure, depth=0) -> str:
    """Format tree structure for LLM navigation.
    Shows titles, summaries, and node_ids - but NOT full text.
    This is what PageIndex calls 'get_document_structure'.
    """
    lines = []
    if isinstance(structure, list):
        for node in structure:
            lines.extend(_format_node(node, depth))
    elif isinstance(structure, dict):
        lines.extend(_format_node(structure, depth))
    return "\n".join(lines)


def _format_node(node, depth) -> list:
    indent = "  " * depth
    lines = []
    nid = node.get("node_id", "?")
    title = node.get("title", "Untitled")
    summary = node.get("summary", node.get("prefix_summary", ""))
    line_num = node.get("line_num", "")

    label = f"{indent}[{nid}] {title}"
    if summary:
        label += f" -- {summary[:120]}"
    if line_num:
        label += f" (line {line_num})"
    lines.append(label)

    if "nodes" in node and node["nodes"]:
        for child in node["nodes"]:
            lines.extend(_format_node(child, depth + 1))
    return lines


def get_node_text(structure, target_ids: list) -> str:
    """Retrieve text content for specific node IDs.
    This is what PageIndex calls 'get_page_content'.
    """
    found = []
    _collect_nodes(structure, target_ids, found)
    return "\n\n---\n\n".join(
        n.get("text", "") for n in found if n.get("text")
    )[:3500]


def _collect_nodes(structure, target_ids, found):
    if isinstance(structure, list):
        for node in structure:
            _collect_nodes(node, target_ids, found)
    elif isinstance(structure, dict):
        nid = str(structure.get("node_id", ""))
        if nid in target_ids:
            found.append(structure)
        if "nodes" in structure:
            _collect_nodes(structure["nodes"], target_ids, found)


def get_all_node_ids(structure) -> list:
    """Get all node IDs in the tree."""
    ids = []
    if isinstance(structure, list):
        for node in structure:
            ids.extend(get_all_node_ids(node))
    elif isinstance(structure, dict):
        if "node_id" in structure:
            ids.append(str(structure["node_id"]))
        if "nodes" in structure:
            ids.extend(get_all_node_ids(structure["nodes"]))
    return ids


# -- Prompts ----------------------------------------------------------------

NAVIGATE_PROMPT = """\
You are navigating a document tree to find information relevant to a question.
Below is the document structure showing section titles and summaries.
Select the node(s) most likely to contain the answer.

Document Structure:
{tree_structure}

Question: {question}

Reply with the node ID(s) that are most relevant, separated by commas.
Example: 0003, 0005
Node IDs:"""

ANSWER_PROMPT = """\
Answer the question using only the retrieved document sections below.
Be concise and specific with numbers and facts.
If the answer is not present in the sections, reply "NOT FOUND".

Retrieved Sections:
{content}

Question: {question}

Answer:"""


# -- Main -------------------------------------------------------------------

def load_tree(domain: str, doc_id: str) -> dict:
    """Load the pre-built tree for a document."""
    doc_hash = hashlib.md5(doc_id.encode()).hexdigest()[:8]
    tree_path = TREES / f"{domain}_{doc_hash}.json"
    if tree_path.exists():
        return json.load(open(tree_path, encoding="utf-8"))
    return None


def run_pageindex_qa(client, question: str, tree: dict, model: str = "gpt-4.1-mini") -> dict:
    """Run one PageIndex QA: navigate tree, retrieve content, answer."""
    structure = tree.get("structure", [])
    if not structure:
        return {"pred_answer": "NOT FOUND", "nav_ids": [],
                "method": "empty_tree", "tokens_in": 0, "tokens_out": 0}

    all_ids = get_all_node_ids(structure)

    # If tree has only 1 node, skip navigation and use it directly
    if len(all_ids) <= 1:
        content = get_node_text(structure, all_ids)
        if not content:
            content = json.dumps(structure)[:3000]
        resp = llm_call(client,
            model=model,
            messages=[{"role": "user", "content":
                ANSWER_PROMPT.format(content=content, question=question)}],
            max_tokens=MAX_TOKENS, temperature=0,
        )
        return {
            "pred_answer": resp.choices[0].message.content.strip(),
            "nav_ids": all_ids,
            "method": "single_node",
            "tokens_in": resp.usage.prompt_tokens,
            "tokens_out": resp.usage.completion_tokens,
        }

    # Step 1: Navigate tree (LLM sees structure, NOT text)
    tree_str = format_tree_for_navigation(structure)
    resp_nav = llm_call(client,
        model=model,
        messages=[{"role": "user", "content":
            NAVIGATE_PROMPT.format(tree_structure=tree_str[:2500],
                                   question=question)}],
        max_tokens=50, temperature=0,
    )
    nav_output = resp_nav.choices[0].message.content.strip()
    tok_nav_in = resp_nav.usage.prompt_tokens
    tok_nav_out = resp_nav.usage.completion_tokens

    # Parse selected node IDs
    selected_ids = re.findall(r'\d{4}', nav_output)
    if not selected_ids:
        # Fallback: try to match any number
        selected_ids = re.findall(r'\d+', nav_output)
    # Pad to 4 digits to match VectifyAI format
    selected_ids = [s.zfill(4) for s in selected_ids]
    # Filter to valid IDs
    selected_ids = [s for s in selected_ids if s in all_ids]
    if not selected_ids:
        selected_ids = all_ids[:3]  # fallback to first 3 nodes

    # Step 2: Retrieve content for selected nodes
    content = get_node_text(structure, selected_ids)
    if not content:
        content = get_node_text(structure, all_ids[:3])

    # Step 3: Answer from retrieved content
    resp_ans = llm_call(client,
        model=model,
        messages=[{"role": "user", "content":
            ANSWER_PROMPT.format(content=content, question=question)}],
        max_tokens=MAX_TOKENS, temperature=0,
    )

    return {
        "pred_answer": resp_ans.choices[0].message.content.strip(),
        "nav_ids": selected_ids,
        "nav_output": nav_output[:200],
        "method": "tree_navigation",
        "tokens_in": tok_nav_in + resp_ans.usage.prompt_tokens,
        "tokens_out": tok_nav_out + resp_ans.usage.completion_tokens,
    }


def run_domain(domain: str):
    out_path = OUT_DIR / f"{domain}.jsonl"
    done_ids = set()
    if out_path.exists():
        done_ids = {json.loads(l)["id"] for l in open(out_path)}

    records = [json.loads(l) for l in open(DATA / domain / "qa_pairs.jsonl")]
    remaining = [r for r in records if r["id"] not in done_ids]

    if not remaining:
        rows = [json.loads(l) for l in open(out_path)]
        em_avg = sum(r["em"] for r in rows) / len(rows) if rows else 0
        print(f"  {domain}: already complete (n={len(rows)}, EM={em_avg:.3f})")
        return

    client = get_client()
    model = DOMAIN_MODELS[domain]
    print(f"  {domain}: model={model}, {len(done_ids)} done, "
          f"running {len(remaining)} remaining...", flush=True)

    for i, r in enumerate(remaining):
        # Rate pacing: 15 req/min, PageIndex uses 1-2 calls per Q
        time.sleep(5)
        t0 = time.time()
        doc_id = r.get("context_doc", r["id"])

        tree = load_tree(domain, doc_id)
        if tree is None:
            tree = {
                "structure": [{"title": doc_id, "node_id": "0001",
                              "text": r.get("context_text", "")[:3000]}],
                "method": "fallback_no_tree",
            }

        result_data = run_pageindex_qa(client, r["question"], tree, model=model)
        latency = round(time.time() - t0, 3)

        result = {
            "id":          r["id"],
            "domain":      domain,
            "question":    r["question"],
            "gold_answer": r["answer"],
            "pred_answer": result_data["pred_answer"],
            "nav_ids":     result_data.get("nav_ids", []),
            "nav_output":  result_data.get("nav_output", ""),
            "em":          exact_match(result_data["pred_answer"], r["answer"]),
            "f1":          round(f1(result_data["pred_answer"], r["answer"]), 4),
            "tokens_in":   result_data.get("tokens_in", 0),
            "tokens_out":  result_data.get("tokens_out", 0),
            "latency_s":   latency,
            "method":      result_data.get("method", "pageindex"),
            "tree_method": tree.get("method", "unknown"),
            "model":       model,
        }
        with open(out_path, "a") as f_out:
            f_out.write(json.dumps(result) + "\n")

        if (i + 1) % 20 == 0:
            print(f"    [{i+1}/{len(remaining)}] last answer: "
                  f"{result_data['pred_answer'][:60]}...")

    all_rows = [json.loads(l) for l in open(out_path)]
    em_avg = sum(r["em"] for r in all_rows) / len(all_rows)
    f1_avg = sum(r["f1"] for r in all_rows) / len(all_rows)
    nf = sum(1 for r in all_rows
             if "NOT FOUND" in r["pred_answer"].upper())
    print(f"  {domain}: EM={em_avg:.3f}  F1={f1_avg:.3f}  "
          f"NOT_FOUND={nf}/{len(all_rows)} ({100*nf/len(all_rows):.1f}%)  "
          f"n={len(all_rows)}")


if __name__ == "__main__":
    print("=== Step 5e: PageIndex Evaluation (Official VectifyAI) ===\n")
    print(f"  Models: {DOMAIN_MODELS}")
    print(f"  Trees: {TREES}")
    print(f"  Output: {OUT_DIR}\n")

    for domain in DOMAINS:
        run_domain(domain)

    print("\nStep 5e complete.")
