import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# faiss must be imported after torch (via sentence_transformers): the other
# order crashes model encoding on macOS (conflicting OpenMP runtimes).
import faiss  # noqa: E402

from retrieval.common import (
    add_query_arguments,
    file_sha1,
    load_chunks,
    load_config,
    make_results,
    output_results,
)


class DenseRetriever:
    """
    Dense retrieval: L2-normalized sentence-transformer embeddings in a FAISS
    inner-product index, so scores are cosine similarities.

    The index is cached under index_dir and reused while the chunk file and
    model settings are unchanged.
    """

    def __init__(
        self,
        chunks: list[dict],
        chunks_path: str | Path,
        model_name: str,
        max_seq_length: int,
        batch_size: int,
        device: str,
        index_dir: str | Path,
        rebuild: bool = False,
    ):
        self.chunks = chunks
        self.model = SentenceTransformer(model_name, device=device)
        # Longer inputs are truncated to this many word pieces, for chunks
        # and queries alike.
        self.model.max_seq_length = max_seq_length

        self.settings = {
            "model_name": model_name,
            "max_seq_length": max_seq_length,
            "chunks_sha1": file_sha1(chunks_path),
            "n_chunks": len(chunks),
        }

        slug = re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_")
        self.cache_dir = Path(index_dir) / f"{slug}_len{max_seq_length}"
        self.index_path = self.cache_dir / "index.faiss"
        self.meta_path = self.cache_dir / "meta.json"

        self.loaded_from_cache = not rebuild and self._cache_is_valid()
        if self.loaded_from_cache:
            self.index = faiss.read_index(str(self.index_path))
            with self.meta_path.open("r", encoding="utf-8") as f:
                self.meta = json.load(f)
        else:
            self._build(batch_size)

    @classmethod
    def from_config(
        cls, config: dict, chunks: list[dict] | None = None, rebuild: bool = False
    ) -> "DenseRetriever":
        settings = config["dense"]
        return cls(
            chunks if chunks is not None else load_chunks(config["chunks_path"]),
            chunks_path=config["chunks_path"],
            model_name=settings["model_name"],
            max_seq_length=settings["max_seq_length"],
            batch_size=settings["batch_size"],
            device=settings["device"],
            index_dir=settings["index_dir"],
            rebuild=rebuild,
        )

    def _cache_is_valid(self) -> bool:
        if not (self.index_path.exists() and self.meta_path.exists()):
            return False
        with self.meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        return all(meta.get(key) == value for key, value in self.settings.items())

    def _build(self, batch_size: int) -> None:
        texts = [c["text"] for c in self.chunks]

        start = time.perf_counter()
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        seconds = time.perf_counter() - start

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        # How many chunks are longer than the model reads.
        encoded = self.model.tokenizer(texts, truncation=False, verbose=False)
        lengths = [len(ids) for ids in encoded["input_ids"]]

        self.meta = {
            **self.settings,
            "embedding_dim": int(embeddings.shape[1]),
            "encode_seconds": round(seconds, 2),
            "chunks_truncated": sum(n > self.model.max_seq_length for n in lengths),
        }

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2)

    def scores(self, query: str) -> np.ndarray:
        """
        Cosine similarity of every chunk to the query, in chunk order.
        """
        query_embedding = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

        found, indices = self.index.search(query_embedding, self.index.ntotal)
        scores = np.empty(self.index.ntotal, dtype=np.float64)
        scores[indices[0]] = found[0]
        return scores

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        return make_results(self.chunks, self.scores(query), top_k)


def main():
    parser = argparse.ArgumentParser(description="Retrieve chunks with dense embeddings.")
    add_query_arguments(parser)
    parser.add_argument("--rebuild", action="store_true", help="Re-encode chunks even if a cached index exists.")
    args = parser.parse_args()

    config = load_config(args.config)
    retriever = DenseRetriever.from_config(config, rebuild=args.rebuild)

    source = "cached" if retriever.loaded_from_cache else "newly built"
    print(f"Index: {retriever.cache_dir} ({source}); {json.dumps(retriever.meta)}\n")

    results = retriever.search(args.query, args.top_k or config["top_k"])
    output_results(results, args)


if __name__ == "__main__":
    main()
