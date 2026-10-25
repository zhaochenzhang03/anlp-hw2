import argparse

import numpy as np

from retrieval.bm25 import BM25Retriever
from retrieval.common import (
    add_query_arguments,
    load_chunks,
    load_config,
    make_results,
    output_results,
    top_indices,
)
from retrieval.dense import DenseRetriever


METHODS = ("bm25", "dense", "rrf", "weighted")


def minmax(scores: np.ndarray) -> np.ndarray:
    low, high = scores.min(), scores.max()
    if high == low:
        return np.zeros_like(scores)
    return (scores - low) / (high - low)


def rrf_fuse(score_lists: list[np.ndarray], k: int = 60, window: int = 100) -> np.ndarray:
    """
    Reciprocal Rank Fusion: each retriever adds 1 / (k + rank) for the chunks
    in its top `window`; other chunks get nothing from it.
    """
    fused = np.zeros(len(score_lists[0]))
    for scores in score_lists:
        for rank, index in enumerate(top_indices(scores, window), start=1):
            fused[index] += 1.0 / (k + rank)
    return fused


def weighted_fuse(sparse: np.ndarray, dense: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    alpha * dense + (1 - alpha) * sparse, after min-max normalizing each
    retriever's scores over all chunks.
    """
    return alpha * minmax(dense) + (1 - alpha) * minmax(sparse)


def fuse(
    method: str,
    sparse: np.ndarray,
    dense: np.ndarray,
    alpha: float = 0.5,
    rrf_k: int = 60,
    rrf_window: int = 100,
) -> np.ndarray:
    if method == "bm25":
        return sparse
    if method == "dense":
        return dense
    if method == "rrf":
        return rrf_fuse([sparse, dense], k=rrf_k, window=rrf_window)
    if method == "weighted":
        return weighted_fuse(sparse, dense, alpha)
    raise ValueError(f"Unknown method: {method}")


class HybridRetriever:
    """
    Combines BM25 and dense retrieval over the same chunks.
    """

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, rrf_k: int = 60, rrf_window: int = 100):
        self.bm25 = bm25
        self.dense = dense
        self.chunks = bm25.chunks
        self.rrf_k = rrf_k
        self.rrf_window = rrf_window

    @classmethod
    def from_config(cls, config: dict) -> "HybridRetriever":
        chunks = load_chunks(config["chunks_path"])
        return cls(
            BM25Retriever.from_config(config, chunks),
            DenseRetriever.from_config(config, chunks),
            rrf_k=config["hybrid"]["rrf_k"],
            rrf_window=config["hybrid"]["rrf_window"],
        )

    def scores(self, query: str, method: str = "rrf", alpha: float = 0.5) -> np.ndarray:
        sparse = self.bm25.scores(query) if method != "dense" else None
        dense = self.dense.scores(query) if method != "bm25" else None
        return fuse(method, sparse, dense, alpha, self.rrf_k, self.rrf_window)

    def search(self, query: str, top_k: int = 5, method: str = "rrf", alpha: float = 0.5) -> list[dict]:
        return make_results(self.chunks, self.scores(query, method, alpha), top_k)


def main():
    parser = argparse.ArgumentParser(description="Retrieve chunks with BM25, dense, or hybrid retrieval.")
    add_query_arguments(parser)
    parser.add_argument("--method", choices=METHODS, default="rrf", help="Retrieval method.")
    parser.add_argument("--alpha", type=float, default=None, help="Dense weight for --method weighted.")
    parser.add_argument("--rrf-k", type=int, default=None, help="RRF constant.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.rrf_k is not None:
        config["hybrid"]["rrf_k"] = args.rrf_k

    retriever = HybridRetriever.from_config(config)
    alpha = args.alpha if args.alpha is not None else config["hybrid"]["default_alpha"]

    results = retriever.search(args.query, args.top_k or config["top_k"], args.method, alpha)
    output_results(results, args)


if __name__ == "__main__":
    main()
