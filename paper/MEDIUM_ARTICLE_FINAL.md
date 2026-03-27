# I Ran 600 Questions Through RAG and a Tree-Based Alternative. RAG's Problem Isn't Accuracy.

If you've spent any time building RAG pipelines, you know the feeling. You point your system at a 200-page 10-K filing and ask "What was Walmart's days payable outstanding in FY2018?" and it says it can't find the answer. The number is right there on page 47, in a table. But the embedding for your question lands nowhere near the chunk containing that table row.

PageIndex, a system from VectifyAI, claims to solve this. Instead of vector search, it builds a tree over the document — paragraph chunks at the leaves, LLM-generated summaries at the parent nodes, a root that captures the whole thing. The LLM walks this tree top-down at query time, picking which branch to follow. They report 98.7% accuracy on FinanceBench.

98.7%. On one benchmark. Evaluated by the team that built it.

I was curious enough to test it myself.

---

## What I Did

Three retrieval systems. Four domains. 150 questions each. Same LLM generates every answer — the only variable is how context gets retrieved.

The systems:
- **BM25**: Old-school keyword matching. No neural anything.
- **Dense RAG**: Sentence embeddings into FAISS, top-5 retrieval. The standard pipeline everyone uses.
- **Tree retrieval**: I replicated PageIndex's algorithm — chunk the doc, build a summary tree with the LLM, navigate it top-down at query time.

The domains: financial reports (FinanceBench), legal contracts (CUAD), scientific papers (QASPER), and technology passages (SQuAD filtered for tech content). A mix of long and short, structured and messy.

Why the same LLM everywhere? Because if system A uses GPT-4 and system B uses GPT-3.5, you're measuring the model difference, not the retrieval difference. I needed to isolate retrieval.

---

## The Obvious Story

[IMAGE: fig1_accuracy_by_domain.png]
*Accuracy by domain (0–4 judge score, higher is better). The stars mark statistically significant differences.*

Finance: tree retrieval blows everything away. 3.94 vs 1.69 for RAG. Huge gap.

Technology: RAG blows tree retrieval away. 3.93 vs 1.75. Equally huge gap, opposite direction.

Legal and science: nothing works. All three systems score below 0.50.

Overall across all 600 questions? No statistically significant difference. p=0.58. A tie.

I nearly wrote this up as "different tools for different jobs" and moved on. Then I looked at the NOT FOUND rates.

---

## What's Actually Going On

This is the table that changed the whole story for me.

Dense RAG said "NOT FOUND" on **53% of financial questions**. It couldn't retrieve any relevant chunk for more than half the questions. Every NOT FOUND scores zero. That's what tanks the average.

Tree retrieval said NOT FOUND on **0%** of financial questions. Zero. It always finds *something* in the tree — it has to, because the LLM picks a branch at each level and always lands on a leaf.

So I filtered out the NOT FOUND cases. Compared only the questions where RAG actually produced an answer.

RAG's score on those 70 questions: **3.63**.

Tree retrieval on the same 70 questions: **3.94**.

The gap goes from 2.25 to 0.31. That's not "tree retrieval is way more accurate." That's "tree retrieval answers more questions."

It flips on technology. Tree retrieval says NOT FOUND 55% of the time on tech passages. RAG answers 99.3%. But when tree retrieval does answer? It scores 3.87 vs RAG's 3.95. Basically the same.

**The systems don't differ in how well they answer. They differ in what they can find.**

---

## The Coverage-Precision Tradeoff

I keep coming back to this framing because it's the thing I wish someone had told me before I started the experiment.

Tree retrieval gives you coverage. The LLM always ends up at a leaf node with text. Maybe it's the right text, maybe it's not. But you always get an answer. This is great for long, structured documents — financial reports with clear sections, regulatory filings, thick technical manuals — where the tree's structure matches the document's structure.

Dense RAG gives you precision. When it finds something, it's usually right. When nothing is close enough, it honestly says so. This works well for short passages where embedding similarity can actually cover the relevant content. FAQ pages, knowledge base articles, anything where one chunk contains the whole answer.

Neither is better. They're solving different problems.

---

## Document Length Is the Deciding Factor

[IMAGE: fig4_length_analysis.png]
*How the tree-vs-RAG gap changes with document length. Positive = tree retrieval wins.*

I split each domain into thirds by document length and computed the gap for each. In three of four domains, longer documents push the needle toward tree retrieval. Technology goes the other way because the passages are so short (average 812 characters) that there's nothing for a tree to add.

This is the clearest practical signal from the whole experiment. If your documents are long and structured, tree retrieval's coverage advantage will probably dominate. If they're short, don't bother with the tree overhead.

---

## Cost

[IMAGE: fig2_token_cost.png]
*Tokens per question by domain and system.*

Tree retrieval uses fewer tokens per question because it selects one node instead of concatenating five chunks. But that's misleading. What you actually care about is tokens per *correct* answer.

On finance: 506 tokens per correct answer for tree retrieval vs 1,683 for RAG. Three times cheaper.

On technology: 765 for RAG vs 1,281 for tree retrieval. Almost twice cheaper.

Cost follows accuracy follows coverage. It all ties back to the same thing.

---

## What I'd Do Differently

A few things I'd want to flag if someone at my company came to me asking "should we switch from RAG to PageIndex?"

First, I didn't use PageIndex's actual code. I reimplemented their algorithm from the paper. My trees might be worse than theirs. Or better. I don't know. Someone should run this comparison with their official implementation.

Second, my embedding model is from 2021. All-MiniLM-L6-v2. 22 million parameters. Nobody ships that in production anymore. Modern embeddings — BGE-large, E5-mistral, GTE-Qwen2 — would almost certainly improve RAG's coverage on financial docs. That 53% NOT FOUND rate could drop a lot with better embeddings. I tested the architecture, not the best available implementation of each architecture.

Third, the tech domain was too easy. SQuAD passages averaging 812 characters — of course dense retrieval works when the passage fits in one chunk. I'd want to rerun this on actual long-form technical documentation (Azure docs, AWS whitepapers, something with real depth) before drawing conclusions about tech content.

Fourth, legal and science were basically failure domains. All three systems scored below 0.50. That means 1000-character chunks with top-5 retrieval just aren't enough for 40-page contracts or multi-section research papers. I didn't learn anything useful about RAG vs trees from those domains — I just learned that my setup was too basic for those tasks.

---

## The Bottom Line

Accuracy benchmarks for RAG systems are mostly measuring retrieval coverage, not answer quality. When both tree retrieval and dense RAG manage to find relevant content, they answer about equally well. The question isn't "which is more accurate?" It's "which documents can each system actually handle?"

Long and structured → tree retrieval.
Short and dense → vector RAG.

Everything else is implementation details.

---

All experiment code and data: [GitHub link]
Full paper with stats: [SSRN link]

*Shivam Rawat is a software engineer at Microsoft Azure. This research was conducted independently and doesn't represent Microsoft's views. Reach him at shivarawat24@gmail.com.*
