#!/usr/bin/env bash
# Core end-to-end ablations on the public dev set (157 questions). Each run
# changes one variable relative to the default (weighted fusion, alpha 0.75,
# top-5, Qwen3-4B-Instruct-2507) and writes outputs/rag/<run-name>/.
# Runs resume where they stopped, so the script can simply be re-run.
#
#   bash scripts/run_ablations.sh                 # all runs
#   EXTRA="--batch-size 8" bash scripts/run_ablations.sh   # e.g. on a GPU
set -euo pipefail

PY=${PY:-python}
EXTRA=${EXTRA:-}

# A. Closed-book vs B-E. retrieval methods (top-5, primary reader)
$PY -m scripts.run_rag_eval --method closed_book $EXTRA
$PY -m scripts.run_rag_eval --method bm25 --top-k 5 $EXTRA
$PY -m scripts.run_rag_eval --method dense --top-k 5 $EXTRA
$PY -m scripts.run_rag_eval --method rrf --top-k 5 $EXTRA
$PY -m scripts.run_rag_eval --method weighted --alpha 0.75 --top-k 5 $EXTRA

# 1. top-k: 10 instead of 5 (weighted 0.75, primary reader)
$PY -m scripts.run_rag_eval --method weighted --alpha 0.75 --top-k 10 $EXTRA

# 2. reader: smaller model (weighted 0.75, top-5)
$PY -m scripts.run_rag_eval --method weighted --alpha 0.75 --top-k 5 --model Qwen/Qwen2.5-1.5B-Instruct $EXTRA
