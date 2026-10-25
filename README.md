# 11-711 HW2: RAG over Pittsburgh and CMU

Retrieval-augmented question answering about Pittsburgh and Carnegie Mellon.
This repository currently covers data collection, chunking, and retrieval
(sparse, dense, hybrid). Answer generation is not implemented yet.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

All commands below run from the repository root with `.venv/bin/python`.

## 1. Corpus

Raw documents live in `data/raw/` (git-ignored; one JSON file per page with
`url`, `title`, `category`, `text`, `retrieved_at`).

```bash
# Crawl every source in scripts/sources.json (failures go to outputs/crawl_failures.json)
python scripts/crawl.py --sources scripts/sources.json

# Carnegie Museum of Art paintings from the museum's CC0 2015 collection data
python scripts/build_cmoa_docs.py
```

## 2. Chunks

```bash
python scripts/chunk.py
```

Writes `data/processed/chunks.jsonl` (committed, so retrieval results are
reproducible without re-crawling). Output is deterministic; chunk IDs are
`<doc_id>#<chunk_index>`.

## 3. Retrieval

All settings are in `configs/retrieval_config.json`:

| Setting | Value |
|---|---|
| Chunks | `data/processed/chunks.jsonl` |
| Text normalization | NFKD, accents stripped, lowercase, tokens `[a-z0-9]+` |
| BM25 | `rank_bm25.BM25Okapi`, k1 = 1.5, b = 0.75, epsilon = 0.25; scikit-learn English stopwords removed; month names shortened ("December" → "dec") in chunks and queries |
| Dense model | `sentence-transformers/all-MiniLM-L6-v2` (384-dim), CPU |
| Dense truncation | `max_seq_length` = 256 word pieces for chunks and queries (chunks longer than that are truncated; 512 tested worse) |
| Dense similarity | L2-normalized embeddings, `faiss.IndexFlatIP` (cosine) |
| RRF | k = 60 over each retriever's top 100 |
| Weighted fusion | `alpha * dense + (1 - alpha) * bm25`, both min-max normalized over all chunks per query; alphas 0.25 / 0.5 / 0.75 |
| Ties | broken by chunk order, so rankings are deterministic |

### Dense index

The FAISS index is built on first use and cached in
`data/index/dense/<model>_len<max_seq_length>/` (git-ignored, ~3.7 MB). It is
reused while the chunk file (SHA-1) and model settings are unchanged, and
rebuilt automatically otherwise. To force a rebuild:

```bash
python -m retrieval.dense --query "test" --rebuild
```

### Retriever CLIs

```bash
python -m retrieval.bm25   --query "What is the name of Carnegie Mellon's business school?" --top-k 5
python -m retrieval.dense  --query "What is the name of Carnegie Mellon's business school?" --top-k 5
python -m retrieval.hybrid --query "..." --method rrf
python -m retrieval.hybrid --query "..." --method weighted --alpha 0.75
python -m retrieval.hybrid --query "..." --method bm25     # or dense
```

Each prints rank, score, chunk ID, title, category, URL and text
(`--max-chars 0` for full text, `--json` for JSON).

## 4. Retrieval evaluation

```bash
python scripts/evaluate_retrieval.py
```

Evaluates BM25, dense, RRF and weighted fusion (each alpha) on
`data/eval/public_evaluation_data.json` at k = 1, 3, 5, 10, 20. There are no
gold chunk IDs, so hits are proxies:

- **exact** (primary): a reference answer, normalized (NFKD, lowercase, month
  names shortened, digits split from letters, punctuation removed), occurs on
  token boundaries in a retrieved chunk.
- **soft**: at least 80% of a reference answer's non-stopword tokens occur in
  one retrieved chunk.

Results go to `outputs/retrieval_eval/`:

- `summary.json`: hit@k per method, overall and by topic; how many answers
  occur anywhere in the corpus (the exact-match ceiling); McNemar tests on hit@5.
- `metrics.csv`: the same numbers in long format.
- `per_question.jsonl`: per question and method, the rank of the first exact
  and soft hit, the top 20 chunk IDs with scores, and how many chunks contain
  the answer (answers found in hundreds of chunks, like "Gift", can produce
  hits from unrelated chunks).

## Notes

- On macOS, `faiss` must be imported after `torch`/`sentence_transformers`;
  the other order crashes encoding. `retrieval/dense.py` imports them in the
  right order.
- No LangChain, LlamaIndex or similar frameworks are used; BM25 comes from
  `rank-bm25`, vector search from FAISS, and the fusion logic is in
  `retrieval/hybrid.py`.
