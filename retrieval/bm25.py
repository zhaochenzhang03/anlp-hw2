import argparse

import numpy as np
from rank_bm25 import BM25Okapi

from retrieval.common import (
    add_query_arguments,
    load_chunks,
    load_config,
    make_results,
    output_results,
    tokenize,
)


class BM25Retriever:
    """
    Sparse retrieval over chunk text with BM25 (Okapi).
    """

    def __init__(
        self,
        chunks: list[dict],
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
        remove_stopwords: bool = True,
        short_months: bool = False,
    ):
        self.chunks = chunks
        self.remove_stopwords = remove_stopwords
        self.short_months = short_months
        corpus = [tokenize(c["text"], remove_stopwords, short_months) for c in chunks]
        self.index = BM25Okapi(corpus, k1=k1, b=b, epsilon=epsilon)

    @classmethod
    def from_config(cls, config: dict, chunks: list[dict] | None = None) -> "BM25Retriever":
        settings = config["bm25"]
        return cls(
            chunks if chunks is not None else load_chunks(config["chunks_path"]),
            k1=settings["k1"],
            b=settings["b"],
            epsilon=settings["epsilon"],
            remove_stopwords=settings["remove_stopwords"],
            short_months=settings.get("short_months", False),
        )

    def scores(self, query: str) -> np.ndarray:
        """
        BM25 score of every chunk for the query, in chunk order.
        """
        tokens = tokenize(query, self.remove_stopwords, self.short_months)
        if not tokens:
            return np.zeros(len(self.chunks))
        return np.asarray(self.index.get_scores(tokens), dtype=np.float64)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        return make_results(self.chunks, self.scores(query), top_k)


def main():
    parser = argparse.ArgumentParser(description="Retrieve chunks with BM25.")
    add_query_arguments(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    retriever = BM25Retriever.from_config(config)
    results = retriever.search(args.query, args.top_k or config["top_k"])
    output_results(results, args)


if __name__ == "__main__":
    main()
