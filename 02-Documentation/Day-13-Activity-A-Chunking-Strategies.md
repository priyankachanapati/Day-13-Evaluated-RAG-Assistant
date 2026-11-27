# Day 13 – Activity A: Chunking Strategies

**Status:** Completed - Pending Review · **Date:** 2026-11-26 · **Phase:** 2 of 9

## 1. Objective

Understand the main families of text splitters, re-index the Day 12 corpus with fixed 500-character chunks and
recursive 800/100 chunks (alongside the existing Day 12 chunker), and compare them with objective measures and with
the top-3 results of five development queries — judged by **evidence coverage**, not only by which policy is returned.

## 2. Splitter families (concepts)

| Family | How it splits | Strength | Weakness | In this activity |
|---|---|---|---|---|
| **Character** | Every N characters, regardless of content | Trivial, predictable sizes | Cuts words, sentences and headings anywhere | `fixed_500` |
| **Recursive character** | Tries separators in order (`\n\n` → `\n` → space → character): split on the coarsest separator present, re-split only pieces still too long, then merge neighbours up to the size limit with an overlap | Respects paragraphs/words when it can; generic | Knows nothing about document semantics; overlap is made of *whole pieces*, so it can be far below the nominal value | `recursive_800_100` |
| **Document-specific** | Uses the format's structure (Markdown headings, paragraphs; code by function; HTML by tag) | Keeps headings with their text, units stay meaningful | Needs a splitter per format | `v1_paragraph_800_150` (Day 12: packs Markdown paragraphs, glues headings to the following paragraph, word-aligned 150-char overlap) |
| **Semantic** | Embeds sentences and starts a new chunk where the similarity between neighbouring sentences drops (topic shift) | Chunks follow topics, not lengths | Needs embeddings at indexing time; sizes vary; thresholds need tuning | Concept only (not required; not implemented) |

## 3. Method

- **Same everything except chunking:** frozen v1 loader and document bodies, all-MiniLM-L6-v2 (ONNX) embeddings
  through v1's `LocalEmbedder`, embedded text = title + chunk (v1's `embedding_text`), cosine distance, v1
  `Retriever`, top-3. One Chroma collection per strategy in `.chroma-v2/` (v1's `.chroma/` untouched).
- **v1 is not re-implemented:** the v1 strategy calls the frozen `chunk_document(800, 150)`; a check confirms it gives
  the same 22 chunks, and that its index reproduces Day 12 run 1's top-3 IDs and 4-dp scores for all 15 smoke questions.
- **Recursive splitter is the standard algorithm:** implemented without the LangChain dependency, then verified
  **byte-identical** to `langchain-text-splitters` 1.2.0 `RecursiveCharacterTextSplitter(800, 100)` on all 6 documents
  (LangChain installed only in a throwaway scratch venv).
- **Canonical evidence units (chunk-independent labels):** every sentence of every numbered section is an evidence
  unit with a stable ID (`HR-001-S02-E04` = HR-001, section 2, sentence 4) and character offsets — 102 units, frozen
  (`evaluation/datasets/evidence_catalog.json`, SHA-256 `2504912c…`, rebuilt byte-identically from the corpus). A chunk
  **contains** a unit only if the **whole** sentence lies inside it.
- **Five development queries** (paraphrase, keyword, identifier, same-section multi-fact, cross-section multi-fact)
  labelled with required evidence **before** any results were produced, frozen at 2026-11-26T15:45:32Z
  (`dev_queries.json`, SHA-256 `9408567a…`). They are development data, never part of the 30-query test set.
- **Offset-based boundary measures** (no visual judgement): mid-sentence boundary, mid-word cut, broken heading,
  trailing heading (chunk ends with a heading), split fact (sentence not intact in any chunk), duplicated fact.

## 4. Results (actual, from `results/chunking-comparison-report.md`)

### 4.1 Size

| Strategy | Chunks | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| v1_paragraph_800_150 | 22 | 639.4 | 676 | 235 | 798 |
| fixed_500 | 28 | 442.0 | 500 | 54 | 500 |
| recursive_800_100 | 20 | 637.8 | 690 | 117 | 780 |

### 4.2 Are headings, sentences and key facts split?

| Strategy | End mid-sentence | Start mid-sentence | Mid-word cuts | Broken headings | Trailing headings | Facts intact | Facts split | Facts duplicated | Boundaries with overlap · mean overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| v1_paragraph_800_150 | 0 | 1 | 0 | 0 | 0 | 102/102 | 0 | 17 | 16/16 · 105.6 chars |
| fixed_500 | 19 | 19 | 22 | 6 | 0 | 83/102 | **19** | 0 | 0/22 · 0 |
| recursive_800_100 | 0 | 0 | 0 | 0 | **12** | 102/102 | 0 | 1 | 12/14 · 27.4 chars |

What this means:
- **fixed_500** cuts through 19 sentences, 22 words and 6 headings; **19 of 102 facts exist in no chunk intact** —
  including "Full-time employees receive 20 days of paid annual leave per calendar year." (`HR-001-S02-E01`, cut after
  "Full-time") and the 15-minute account lockout rule. No retrieval method can return those facts whole from this index.
- **recursive_800_100** never cuts a sentence or word, but **12 chunks end with a bare heading** (e.g. `## 4. Sick
  Leave`), separated from its text. The heading is then repeated as the next chunk's overlap: with paragraph-sized
  pieces (> 100 chars), the only piece that fits the 100-character overlap is the heading. Mean real overlap is
  27.4 characters; 2 of 14 boundaries have none. "Overlap 100" is an upper bound, not a guarantee.
- **v1_paragraph_800_150** keeps every fact and heading intact; its one mid-sentence start is an overlap tail
  (`HR-003-CH04` starts "beverages, traffic or parking…", whose full sentence is intact in `HR-003-CH03`). The larger
  overlap duplicates 17 facts in two chunks — harmless for answering, but it means **counting relevant chunks would
  over-count**; Phase 5 metrics therefore count distinct evidence IDs.
- Section labels differ in meaning: v1 labels a chunk with the sections of its *new* (non-overlap) text, the new
  strategies with every section the chunk overlaps. Evidence containment, not the label, is used for scoring.

### 4.3 Development queries (top-3)

| Query (category) | v1_paragraph_800_150 | fixed_500 | recursive_800_100 |
|---|---|---|---|
| DEV-01 two-week holiday notice (paraphrase) | policy ✔, evidence ✘ (rank 4) | policy ✔, evidence ✘ (rank 4) | policy ✔, evidence ✔ rank 1 |
| DEV-02 KeyVault (keyword) | ✔ rank 1 | ✔ rank 1 | ✔ rank 1 |
| DEV-03 Service Desk extension (identifier) | ✔ rank 1 | ✔ rank 1 | ✔ rank 1 |
| DEV-04 Wi-Fi + VPN (2 facts, same section) | 2/2, rank 1 | 2/2, rank 1 | 2/2, rank 1 |
| DEV-05 premium economy + meals (2 facts, 2 sections) | 2/2, rank 1 | 2/2, rank 1 | 2/2, rank 1 |
| **Total** | policy 5/5 · evidence hit 4/5 · groups 6/7 | policy 5/5 · evidence hit 4/5 · groups 6/7 | policy 5/5 · evidence hit 5/5 · groups 7/7 |

- **Policy-level matching cannot tell the strategies apart (5/5 each); evidence coverage can.** In DEV-01 all three
  return HR-001 chunks, but for v1 and fixed_500 none of the top-3 contains the needed sentence.
- DEV-01 diagnostic (full ranking, `03-Evidence/10-…`): the needed sentence is at **rank 4** for v1 and fixed_500.
  Both rank first a chunk containing the *Public Holidays* section — the query word "holiday" pulls it up although
  the user means annual leave (vocabulary mismatch). A hypothesis to test in Activities B/C, not a result.
- Similarity values are **not** a quality comparison across strategies: the chunk texts differ, so scores differ.

## 5. Conclusion — no strategy selected

The measurements show clear structural trade-offs: fixed_500 breaks 19 facts; recursive_800_100 keeps all facts but
separates 12 headings from their text; v1 keeps facts and headings intact at the cost of duplicated text. On the five
dev queries recursive_800_100 covered one more evidence group than the others — **a one-query difference on a
development sample is not evidence of a better strategy**, and dev queries are for tuning, not for selection.
As agreed, **no winner is selected in Activity A**. All three indexes remain available for hybrid retrieval (B) and
reranking (C), and the chunking used by RAG v2 will be chosen from the 30-query held-out evaluation (Phase 7).

## 6. Reproducibility and verification

| Check | Result | Evidence |
|---|---|---|
| Integrity gate (corpus, catalog, dev-query hashes; catalog rebuild; v1 strategy == frozen v1) | PASS | report §0 |
| v1-strategy index reproduces Day 12 run 1 (15 smoke questions, top-3 IDs + 4-dp scores) | PASS 15/15 | report §0 |
| Recursive splitter vs LangChain 1.2.0 | 6/6 documents byte-identical | `03-Evidence/06-…` |
| Two full runs (indexes rebuilt) | identical results SHA-256 `0c6e3c18…`; logs identical except timestamp | `03-Evidence/07-…`, `08-…` |
| Tests | 67 passed, 2 deselected (42 frozen v1 + 25 new: hand-worked splitter examples, lossless/size/overlap properties, v1 equivalence, catalog, containment, boundary analysis) | `03-Evidence/09-…` |
| Frozen baseline after Phase 2 | manifests PASS; Phase 1 reproduction PASS (60/60 exact); Day 12 git clean | `03-Evidence/09-…` |

## 7. Issues encountered

- One of my hand-calculated test expectations was wrong (`"ccc "` + `"ddd"` = 7 characters fits a 7-character
  chunk); the code was right. Test expectation corrected with the arithmetic written in the test.

## 8. Files

`rag_v2/chunking.py`, `rag_v2/indexing.py`, `rag_v2/chunk_analysis.py`, `evaluation/evidence.py`,
`scripts/build_evidence_catalog.py`, `evaluation/datasets/{evidence_catalog.json, dev_queries.json}`,
`manifests/{evidence-catalog,dev-queries}.sha256`, `activities/a-chunking-strategies/{compare_chunking.py, README.md,
results/}`, `tests_v2/`. DOCX version for the Google Doc: [PENDING – Phase 9].
