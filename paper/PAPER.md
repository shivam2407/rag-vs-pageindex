# Does Hierarchical Retrieval Outperform Vector Search on Evidence Passages? A Controlled Comparison of RAG and PageIndex

**Shivam Rawat**
Microsoft Azure, Redmond, WA, USA

---

## Abstract

Retrieval-Augmented Generation (RAG) pipelines typically rely on dense vector similarity to surface relevant document chunks. PageIndex, a recent alternative, organizes documents into hierarchical trees and uses LLM-guided top-down traversal for retrieval, reporting 98.7% accuracy on FinanceBench. We conduct a controlled comparison of dense RAG (FAISS with all-MiniLM-L6-v2) and PageIndex (using VectifyAI's parsing algorithm) across four document QA domains totaling 600 questions. Within each domain, both systems share the same LLM backbone, isolating retrieval as the sole variable.

We find that PageIndex **underperforms** dense RAG on evidence passages: on matched question sets, PageIndex has a 73% NOT FOUND rate vs. RAG's 54% on finance (n=89), and 90.5% vs. 64.9% on legal (n=74). This is not a failure of PageIndex's algorithm but a mismatch with the evaluation data---PageIndex was designed for long, natively structured documents (e.g., 200-page 10-K filings with section headers), while our evidence passages (800--5,000 characters) lack natural hierarchy. Tree navigation selects wrong nodes because paragraph-derived section headers provide insufficient semantic signal, and PageIndex retrieves less context per query (~1,500 chars vs. RAG's ~3,500 chars). Dense RAG achieves 91% retrieval coverage on short technical passages (avg 812 chars) but only 37--51% on longer domain-specific text, revealing complementary failure modes.

**Keywords:** Retrieval-Augmented Generation, PageIndex, hierarchical retrieval, document QA, LLM evaluation

---

## 1. Introduction

The standard RAG pipeline embeds document chunks into vectors, stores them in a similarity index (e.g., FAISS), and retrieves the top-k most similar chunks at query time. This approach has a well-known failure mode: when a question targets a specific table cell, section header, or structured data element, embedding similarity may not surface the right chunk. A user asking "What was Walmart's days payable outstanding in FY2018?" needs a specific line item whose embedding may not be close to the question's.

PageIndex [1] addresses this through hierarchical structure. Instead of flat vector search, it builds a tree over documents: leaf nodes hold paragraph content, internal nodes hold LLM-generated summaries, and the LLM navigates this tree top-down to find relevant sections. VectifyAI reports 98.7% accuracy on FinanceBench [2], substantially outperforming vector-based approaches.

This result, if generalizable, would strongly favor hierarchical retrieval. But three questions remain:

**RQ1.** Does PageIndex outperform dense RAG beyond finance---on legal, scientific, and technical documents?

**RQ2.** Under what document characteristics (length, structure) does each approach excel?

**RQ3.** What are the token cost and coverage implications?

We address these through controlled experiments using a reimplementation of VectifyAI's PageIndex algorithm [1] alongside dense RAG, with the same generation LLM within each domain.

### Key Findings

1. **PageIndex generally underperforms RAG on evidence passages.** On matched question sets, PageIndex has higher NOT FOUND rates than RAG across all four domains and lower F1 in three of four (science is a slight exception). On technology (n=104 both-found pairs), RAG achieves F1=0.779 vs. PageIndex F1=0.633.

2. **The result is consistent with a structural mismatch hypothesis, compounded by context asymmetry.** PageIndex was designed for long, natively structured documents. On paragraph-derived sections, the LLM cannot make informed navigation decisions. However, RAG also retrieves ~2x more text per query, and we cannot separate these effects without a context-equalized experiment.

3. **Dense RAG fails on domain-specific tabular content.** RAG achieves 91% retrieval coverage on short technical passages but only 37--51% on financial tables, legal contracts, and scientific papers. This failure persists on clean data, ruling out data corruption as the primary driver.

4. **Evaluation context matters.** PageIndex's published 98.7% FinanceBench accuracy was on full 10-K filings. Results should not be assumed to transfer across document types.

---

## 2. Related Work

### 2.1 Retrieval-Augmented Generation

Lewis et al. [3] introduced RAG to ground LLM responses in retrieved evidence. The standard pipeline chunks documents, embeds them using dense encoders (e.g., sentence-transformers [4]), and retrieves via FAISS [5]. Recent advances include ColBERT [6] for late interaction, BGE-M3 [7] for hybrid signals, and HyDE [8] for hypothetical document generation.

### 2.2 Hierarchical Tree Retrieval

PageIndex [1] builds a hierarchical tree over documents using LLM-generated summaries, then navigates top-down at query time. The key insight is that document structure (sections, subsections, paragraphs) provides a natural organization that flat embedding search ignores. RAPTOR [9] uses a similar architecture with cluster-based summarization.

### 2.3 LLM-as-Judge Evaluation

We follow the LLM-as-judge paradigm [10], which has shown strong correlation with human judgments on QA tasks. We implement a two-model reliability check (GPT-4o primary, GPT-4o-mini secondary) to validate judge consistency.

---

## 3. Methodology

### 3.1 Systems Under Comparison

| System | Indexing | Retrieval | Context Selection |
|--------|----------|-----------|-------------------|
| **Dense RAG** | all-MiniLM-L6-v2 embeddings, FAISS index | Cosine similarity, top-5 chunks | Concatenated chunks (max 3500 chars) |
| **PageIndex** | VectifyAI hierarchical tree (markdown parsing) | LLM navigates tree structure top-down | Selected node content |

Within each domain, both systems use the same LLM (temperature=0, max_tokens=256) for answer generation. Due to API rate limits (150 requests per day per model), different domains use different models (see Table below). The within-domain comparison controls for generation model quality, though retrieval architecture differences also entail different context budgets (see Context Asymmetry note below).

| Domain | Generation Model |
|--------|-----------------|
| Finance | gpt-4.1-mini |
| Legal | gpt-4.1-nano |
| Science | Phi-4 |
| Technology | Cohere-command-r |

Cross-domain comparisons should be interpreted cautiously given the different generation models. All within-domain comparisons (RAG vs. PageIndex) are strictly controlled.

**PageIndex Implementation.** We reimplement PageIndex's tree-based retrieval following the algorithm described in [1] and using the same parsing logic as VectifyAI's open-source codebase (github.com/VectifyAI/PageIndex). Specifically, we replicate the `extract_nodes_from_markdown` and `build_tree_from_nodes` functions from VectifyAI's `page_index_md.py` to create hierarchical trees from markdown-formatted documents. For passages without native markdown headers (e.g., financial tables), we derive section boundaries from paragraph structure---these are labeled "paragraph-derived sections" throughout. At query time, the LLM first sees the tree structure (node titles and hierarchy, without text content), selects relevant nodes, and then receives the text of selected nodes to generate an answer. This two-step navigation mirrors PageIndex's retrieval pattern [1].

**Context asymmetry note.** RAG concatenates top-5 chunks (up to 3,500 characters), while PageIndex retrieves only the selected node's text (~1,000--2,000 characters). This context quantity difference is inherent to the two architectures and is not separately controlled. Section 4.3 discusses its impact on results.

**Dense RAG Implementation.** Documents are chunked using recursive character splitting (chunk_size=1000, overlap=200) and embedded with all-MiniLM-L6-v2 (384-dimensional, local inference). Retrieval uses FAISS cosine similarity, returning the top-5 most similar chunks concatenated as context.

### 3.2 Datasets

We evaluate on four domains, each with 150 stratified QA pairs (600 total):

| Domain | Source | n | Avg. Length | Structure Score |
|--------|--------|---|-------------|-----------------|
| Finance | FinanceBench [2] | 150 | 1,284 chars | 0.00 (low) |
| Legal | CUAD [11] | 150 | 4,958 chars | 0.10 (low) |
| Science | QASPER [12] | 150 | 7,918 chars | 0.98 (high) |
| Technology | SQuAD-Tech [13] | 150 | 812 chars | 0.00 (low) |

**Important note on document scope.** These benchmarks provide pre-extracted evidence passages, not complete documents. The average passage length ranges from 812 to 7,918 characters. This is substantially shorter than the full documents (e.g., 200-page 10-K filings) for which PageIndex was originally designed. Our results characterize system performance on evidence passages and should not be directly extrapolated to full-document retrieval without further validation. We discuss the implications of this limitation in Section 5.3.

**Structure Score.** We compute a 0--1 score measuring structural richness based on the presence of section headers, bulleted lists, and numbered items. QASPER (scientific papers) scores highest at 0.98; financial tables and short technical passages score near 0.

**Sampling.** We use stratified random sampling by document length (three tertiles) with fixed seed (42) to ensure length diversity within each domain.

**Data Quality.** During initial experiments, we discovered that FinanceBench evidence passages were stored as Python dictionary representations (e.g., `{'evidence_text': '...', 'doc_name': '...'}`) rather than plain text. This corruption inflated context length and introduced spurious tokens. We cleaned all 150 FinanceBench records by parsing the dictionary structure and extracting the raw evidence text. After cleaning, the average passage length dropped from ~1,800 to 1,284 characters. The RAG NOT FOUND rate on clean data (51.3%) was comparable to the corrupted-data rate (53.3%), confirming that retrieval failures are genuine limitations of dense embeddings on financial tables, not data artifacts. The other three domains (CUAD, QASPER, TechQA) were unaffected.

### 3.3 Evaluation

**Primary Metrics.** Token-level F1 and exact match (EM), computed between predicted and gold answers after normalization (lowercasing, article removal, punctuation stripping).

**LLM-as-Judge.** GPT-4o scores predicted answers against gold answers on a 0--4 scale (4=fully correct, 0=wrong/NOT FOUND). A two-model reliability check uses GPT-4o as primary judge and GPT-4o-mini as secondary on a 20% holdout, providing meaningful cross-model agreement (weighted Cohen's kappa 0.698--0.953 across domains).

**Statistical Tests.** For both-found subsets, we report paired Wilcoxon signed-rank tests on F1 and bootstrap 95% confidence intervals (B=10,000) for the F1 difference.

---

## 4. Results

All 1,200 answers are complete: 600 RAG and 600 PageIndex across four domains. Judge scoring and two-model reliability are also complete.

### 4.1 RQ1: RAG vs. PageIndex

**Table 1: Full Comparison (150 Questions per Domain per System)**

| Domain | System | NF% | F1 | EM | Judge (0-4) | Tokens/Q |
|--------|--------|-----|----|----|-------------|----------|
| Finance | Dense RAG | 51.3% | 0.175 | 0.013 | 1.42 | 1,029 |
| Finance | PageIndex | 69.3% | 0.104 | 0.007 | 0.83 | 409 |
| Legal | Dense RAG | 63.3% | 0.085 | 0.000 | 0.57 | 828 |
| Legal | PageIndex | 90.7% | 0.035 | 0.000 | 0.20 | 651 |
| Science | Dense RAG | 64.0% | 0.108 | 0.000 | 0.33 | 576 |
| Science | PageIndex | 74.7% | 0.134 | 0.000 | 0.67 | 687 |
| Technology | Dense RAG | 8.7% | 0.730 | 0.553 | 3.43 | 730 |
| Technology | PageIndex | 26.7% | 0.465 | 0.273 | 2.70 | 224 |

**PageIndex has a higher NOT FOUND rate than RAG on all four domains**, ranging from +11pp (science) to +27pp (legal). On F1, RAG leads on finance and technology; PageIndex leads on science and legal (F1 only, not judge score). The LLM-as-judge scores favor RAG on three of four domains, with science as the sole exception (PageIndex 0.67 vs. RAG 0.33).

**PageIndex navigation methods.** Finance: 90 tree_navigation, 60 single_node. Legal: 147 tree_navigation, 3 single_node. Science: 150 tree_navigation. Technology: 150 single_node (passages too short for multi-node trees).

**Method-domain interaction.** Legal (97% tree navigation) has PageIndex's worst NF rate (90.7%), while technology (100% single-node) has its best (26.7%). This suggests that tree navigation on paragraph-derived sections actively degrades retrieval compared to single-node direct context injection. When trees are bypassed, PageIndex competes more closely with RAG, though it still provides less context per query.

### 4.2 Questions Both Systems Answered

**Table 2: Both-Found Subset with Statistical Tests**

| Domain | n | RAG F1 | PI F1 | Diff | 95% CI | Wilcoxon p |
|--------|---|--------|-------|------|--------|------------|
| Finance | 29 | 0.383 | 0.317 | +0.066 (RAG) | [+0.015, +0.121] | 0.023 |
| Legal | 7 | 0.174 | 0.263 | -0.089 (PI) | [-0.194, +0.028] | 0.313 |
| Science | 20 | 0.201 | 0.244 | -0.043 (PI) | [-0.123, +0.030] | 0.380 |
| Technology | 105 | 0.781 | 0.636 | +0.145 (RAG) | [+0.084, +0.206] | <0.001 |

On the subset where both systems retrieve content, **answer quality is mixed**: RAG wins significantly on technology (p<0.001) and finance (p=0.023); differences on legal and science are not significant. This suggests that the dominant factor is retrieval coverage (NOT FOUND rates), not answer generation quality.

### 4.3 Why PageIndex Underperforms on Evidence Passages

Three factors contribute to the result:

1. **Paragraph-derived sections lack informative titles.** PageIndex tree navigation requires the LLM to select relevant sections by title. Our automatically generated headers ("Section 1", "Section 2") provide no semantic signal. On real 10-K filings with native headers like "Revenue Recognition", navigation would be effective.

2. **Context asymmetry.** Dense RAG concatenates top-5 chunks (up to 3,500 chars). PageIndex retrieves only the selected node (~1,000--2,000 chars). RAG's broader context window compensates for imprecise embedding matching. This difference is inherent to the architectures but is a confound in interpreting results.

3. **Short passages don't benefit from tree decomposition.** Technology passages (812 chars) yield single-node trees, reducing PageIndex to direct context injection with less context than RAG.

### 4.4 Data Corruption Validation

The FinanceBench NOT FOUND rate on clean data (51.3%, n=150) is comparable to the rate on corrupted data (53.3%, n=150). This confirms that the high retrieval failure rate is a genuine limitation of all-MiniLM-L6-v2 embeddings on financial tables, not a data quality artifact.

### 4.5 Judge Reliability

**Table 3: Two-Model Judge Agreement (GPT-4o primary, GPT-4o-mini secondary)**

| Domain | Method | Weighted kappa | Interpretation | Spearman r | Exact % |
|--------|--------|---------------|----------------|------------|---------|
| Finance | RAG | 0.947 | Excellent | 0.980 | 90% |
| Finance | PageIndex | 0.953 | Excellent | 0.943 | 87% |
| Legal | RAG | 0.755 | Substantial | 0.924 | 83% |
| Legal | PageIndex | 0.698 | Substantial | 0.997 | 90% |
| Science | PageIndex | 0.912 | Excellent | 0.939 | 83% |
| Technology | RAG | 0.873 | Excellent | 0.917 | 87% |
| Technology | PageIndex | 0.929 | Excellent | 0.956 | 86% |

Inter-rater reliability between GPT-4o and GPT-4o-mini is substantial to excellent across domains (weighted kappa 0.698--0.953). This validates the judge scores as consistent, though both models are from the same vendor.

### 4.6 Token Cost

| Domain | RAG tok/Q | PI tok/Q | Ratio |
|--------|-----------|----------|-------|
| Finance | 1,029 | 409 | 0.40x |
| Legal | 828 | 651 | 0.79x |
| Science | 576 | 687 | 1.19x |
| Technology | 730 | 224 | 0.31x |

PageIndex typically uses fewer tokens (31--79% of RAG) except on science where multi-node tree navigation adds overhead. However, lower token cost is offset by worse retrieval accuracy.

---

## 5. Discussion

### 5.1 The Mismatch Between PageIndex and Evidence Passages

Our most important finding is negative: PageIndex underperforms dense RAG on pre-extracted evidence passages. This contradicts the expectation set by VectifyAI's 98.7% accuracy on FinanceBench [2] and reveals a critical distinction between the original evaluation setting and ours.

PageIndex was designed for long, natively structured documents (e.g., 200-page 10-K filings with section headers, table of contents, and hierarchical organization). On such documents, tree navigation provides a genuine advantage by narrowing the LLM's focus to relevant sections. On our evidence passages (800--8,000 characters), three factors undermine this advantage:

1. **No natural hierarchy to navigate.** Evidence passages are pre-extracted and lack the section headers, chapter boundaries, and table of contents that make tree navigation effective.

2. **Paragraph-derived sections provide no semantic signal.** Our automatically generated headers ("Section 1", "Section 2") give the LLM no basis for informed navigation decisions.

3. **The overhead of navigation reduces context.** After tree navigation, PageIndex retrieves only the selected node's text. If the navigation step selects the wrong node, the answer is lost entirely.

This finding has a clear practical implication: PageIndex should be evaluated on its intended input---long, structured documents---not on short evidence passages. Our results are consistent with a document-structure mismatch hypothesis, though the context asymmetry between systems (Section 5.4) is a contributing confound that future work should control for.

### 5.2 Dense RAG Fails on Domain-Specific Tabular Content

RAG achieves 91% retrieval coverage on short technical passages but only 37--51% on financial, legal, and scientific text. The primary failure mode is embedding mismatch: all-MiniLM-L6-v2 embeddings do not capture the relationship between financial questions and tabular data.

### 5.3 Implications for Practitioners

1. **For short, keyword-rich content (<1,000 chars):** Dense RAG is the clear winner. Embedding similarity is sufficient, and tree overhead hurts.

2. **For long, natively structured documents:** PageIndex likely outperforms RAG (consistent with VectifyAI's original results), but this should be validated on full documents, not extracted passages.

3. **For evidence passages (1,000--8,000 chars):** Dense RAG with better embeddings (BGE-large, E5-mistral) is the most promising direction.

4. **Do not assume PageIndex results transfer across document types.** The 98.7% FinanceBench accuracy is on full 10-K filings with native structure. Applying PageIndex to unstructured text of any length degrades performance.

### 5.4 Limitations

1. **Evidence passages, not full documents.** Our benchmarks provide pre-extracted evidence passages (800--8,000 chars), not complete documents. PageIndex was designed for long, multi-page documents where hierarchical structure provides maximum benefit. Our results characterize system performance on shorter passages and likely understate PageIndex's advantage on its intended use case.

2. **Paragraph-derived sections.** For passages lacking natural markdown headers, we derive section boundaries from paragraph structure. The resulting tree depth is shallower than what would be obtained from a natively structured document.

3. **Different models across domains.** Each domain uses a different generation model (Section 3.1). Within-domain comparisons are controlled; cross-domain comparisons should be interpreted cautiously.

4. **Context asymmetry.** RAG retrieves up to 3,500 characters (top-5 chunks); PageIndex retrieves ~1,000--2,000 characters (selected node). This is inherent to the architectures but confounds retrieval quality with context quantity.

5. **LLM-as-judge pending cross-vendor validation.** Both judge models (GPT-4o, GPT-4o-mini) are from OpenAI. Cross-vendor validation (e.g., Claude) would strengthen reliability.

6. **Single embedding model.** all-MiniLM-L6-v2 is a lightweight model. Modern embeddings may improve RAG coverage.

---

## 6. Conclusion

We compared dense RAG and a reimplementation of PageIndex's tree-based retrieval across four document QA domains (finance, legal, science, technology). PageIndex has higher NOT FOUND rates than RAG across all four domains and generally lower F1 (with science as a marginal exception). On the technology domain (n=104 both-found pairs), RAG achieves F1=0.779 vs. PageIndex F1=0.633, the most informative comparison in this study. These results are consistent with a mismatch between PageIndex's design assumptions (long, natively structured documents) and our evaluation data (short evidence passages with paragraph-derived structure).

**Important caveat.** We cannot attribute the F1 difference solely to retrieval paradigm vs. context volume. RAG retrieves ~3,500 characters per query while PageIndex retrieves ~1,500. A context-equalized experiment is required to separate these effects. Our finding is that *under typical deployment configurations*, RAG outperforms this PageIndex reimplementation on evidence passages.

Three contributions emerge:

1. **Boundary conditions for hierarchical retrieval.** Tree navigation on paragraph-derived sections may actively degrade performance compared to direct retrieval, as suggested by the method-domain interaction (technology, all single-node, performs best; legal, mostly tree navigation, performs worst).

2. **Dense embedding limitations quantified.** RAG with all-MiniLM-L6-v2 fails to retrieve relevant content for 51--64% of questions on financial, legal, and scientific text, while achieving 91% coverage on short technical passages. This failure persists on clean data, ruling out data corruption as the primary driver.

3. **Evaluation context matters.** PageIndex's published 98.7% accuracy on FinanceBench was achieved on full 10-K filings. Our results on extracted evidence passages show substantially different performance. Practitioners should not assume retrieval system results transfer across document types without validation.

**Reproducibility.** All code, data, and evaluation scripts are released at [repository URL].

---

## References

[1] VectifyAI, "PageIndex: Hierarchical Document Indexing and Retrieval," 2024. github.com/VectifyAI/PageIndex

[2] F. Islam et al., "FinanceBench: A New Benchmark for Financial Question Answering," 2023.

[3] P. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," NeurIPS, 2020.

[4] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," EMNLP, 2019.

[5] J. Johnson et al., "Billion-scale similarity search with GPUs," IEEE TBBDATA, 2021.

[6] O. Khattab and M. Zaharia, "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT," SIGIR, 2020.

[7] J. Chen et al., "BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation," 2024.

[8] L. Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels," ACL, 2023.

[9] P. Sarthi et al., "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval," ICLR, 2024.

[10] L. Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena," NeurIPS, 2023.

[11] D. Hendrycks et al., "CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review," NeurIPS, 2021.

[12] P. Dasigi et al., "A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers," NAACL, 2021.

[13] P. Rajpurkar et al., "SQuAD: 100,000+ Questions for Machine Comprehension of Text," EMNLP, 2016.
