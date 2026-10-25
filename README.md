# 11-711 HW2: RAG over Pittsburgh and CMU

Retrieval-augmented question answering about Pittsburgh and Carnegie Mellon.
Pipeline: data collection → chunking → retrieval (sparse, dense, hybrid) →
answer generation with an open instruct model → evaluation and submission
export.

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

# Targeted patch round (CMU, restaurants/food, taxes/budget, events): sources
# in scripts/sources_patch.json. Pages that block crawlers (City of Pittsburgh,
# Allegheny County) are fetched from Wayback Machine copies given as
# "archived_url"; the document keeps the official "url" plus "archived_url".
python scripts/crawl.py --sources scripts/sources_patch.json --failures outputs/crawl_failures_patch.json

# Pittsburgh Pirates 40-man roster (one record per player) from MLB's Stats API
python scripts/build_pirates_roster_docs.py
```

The crawler parses PDFs with `pypdf`, falls back to the whole page body when
`<main>`/`<article>` is nearly empty, and prepends a page's meta description
when it is not already in the text (useful for JavaScript-rendered sites).

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

## 5. Answer generation

Reader: `Qwen/Qwen3-4B-Instruct-2507` (open weights, released 2025-08, 4.0B
parameters, pinned revision in `configs/generation_config.json`), greedy
decoding, at most 32 new tokens. The prompt (`generation/prompts.py`, version
recorded in every prediction) asks for a bare short answer copied exactly
from the numbered context passages, or "I don't know". Default retrieval:
weighted fusion, alpha 0.75, top-5 (`configs/retrieval_config.json`,
`frozen` section).

```bash
# One run over the public dev set; predictions are appended to
# outputs/rag/<run-name>/predictions.jsonl as they are generated and the run
# is scored at the end (outputs/rag/<run-name>/metrics.json).
python -m scripts.run_rag_eval --method weighted --alpha 0.75 --top-k 5
python -m scripts.run_rag_eval --method rrf --top-k 5
python -m scripts.run_rag_eval --method closed_book
python -m scripts.run_rag_eval --model Qwen/Qwen2.5-1.5B-Instruct

# Subsets: --start/--end (indices) or --ids 1,2,3. Re-running the same command
# resumes (completed IDs are skipped); --overwrite starts over.
python -m scripts.run_rag_eval --method rrf --start 0 --end 40

# All core ablations (closed-book, BM25, dense, RRF, weighted; top-10; small reader)
bash scripts/run_ablations.sh

# Score, analyze failures, build the results tables
python -m scripts.score_rag outputs/rag/*/predictions.jsonl
python -m scripts.error_analysis outputs/rag/weighted0.75_k5__Qwen3-4B-Instruct-2507/predictions.jsonl
python -m scripts.make_results_tables
```

Answer metrics (`evaluation/metrics.py`), each the maximum over a question's
reference answers, after SQuAD normalization (lowercase, no punctuation or
articles): exact match, token F1, answer recall (share of reference tokens in
the answer) and ROUGE-L F1. The course's LLM-as-judge score is not
reproduced locally.

### Running on Colab (GPU)

```bash
!git clone https://github.com/zhaochenzhang03/anlp-hw2.git
%cd anlp-hw2
!pip install -q rank-bm25==0.2.2 faiss-cpu "sentence-transformers>=3" "transformers>=4.51" accelerate pypdf
# Persist outputs on Drive so a disconnect loses nothing:
from google.colab import drive; drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/anlp-hw2-outputs
!python -m scripts.run_rag_eval --method weighted --alpha 0.75 --top-k 5 --batch-size 8 \
    --output-dir /content/drive/MyDrive/anlp-hw2-outputs
# After a disconnect, run the same command again: it resumes.
!EXTRA="--batch-size 8 --output-dir /content/drive/MyDrive/anlp-hw2-outputs" bash scripts/run_ablations.sh
```

On Colab, install packages individually as above rather than
`pip install -r requirements.txt`, which pins numpy/scipy versions built for
Python 3.14 and would replace Colab's preinstalled builds.

## 6. Submission files

```bash
# Unseen test set: {"1": "...", ...}
python -m scripts.run_rag_eval --questions data/eval/<test_queries>.json --run-name test_system1
python -m scripts.export_system_output --predictions outputs/rag/test_system1/predictions.jsonl \
    --questions data/eval/<test_queries>.json --output system_outputs/system_output_1.json

# Leaderboard: {"andrewid": "...", "1": "...", ...}
python -m scripts.run_rag_eval --questions data/eval/leaderboard_queries.json --run-name leaderboard_system1
python -m scripts.export_system_output --predictions outputs/rag/leaderboard_system1/predictions.jsonl \
    --questions data/eval/leaderboard_queries.json --andrewid <ANDREWID> --output outputs/leaderboard/system1.json
```

The exporter refuses to write a file with missing, empty or multi-line
answers, prompt text, or IDs out of order.

## Tests

```bash
python -m unittest discover tests
```

## Notes

- On macOS, `faiss` must be imported after `torch`/`sentence_transformers`;
  the other order crashes encoding. `retrieval/dense.py` imports them in the
  right order.
- No LangChain, LlamaIndex or similar frameworks are used; BM25 comes from
  `rank-bm25`, vector search from FAISS, and the fusion logic is in
  `retrieval/hybrid.py`.
