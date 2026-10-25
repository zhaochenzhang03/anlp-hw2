import hashlib
import json
import re
import textwrap
import unicodedata
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


DEFAULT_CONFIG_PATH = Path("configs/retrieval_config.json")

TOKEN_RE = re.compile(r"[a-z0-9]+")

# Full month names and their short forms, so "December 10" and "Dec 10"
# compare equal.
MONTHS = {
    "january": "jan", "february": "feb", "march": "mar", "april": "apr",
    "june": "jun", "july": "jul", "august": "aug", "september": "sep",
    "sept": "sep", "october": "oct", "november": "nov", "december": "dec",
}
MONTH_RE = re.compile(r"\b(" + "|".join(MONTHS) + r")\b")


def load_config(path: str | Path | None = None) -> dict:
    with Path(path or DEFAULT_CONFIG_PATH).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_chunks(path: str | Path) -> list[dict]:
    """
    Load chunks from a JSONL file, in file order.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def file_sha1(path: str | Path) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    """
    Lowercase and strip accents, e.g. "Matisse’s Café" -> "matisse’s cafe".
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower()


def canonical_months(text: str) -> str:
    """
    Replace full month names in normalized text with their short forms.
    """
    return MONTH_RE.sub(lambda m: MONTHS[m.group(1)], text)


def tokenize(text: str, remove_stopwords: bool = True, short_months: bool = False) -> list[str]:
    text = normalize_text(text)
    if short_months:
        text = canonical_months(text)
    tokens = TOKEN_RE.findall(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    return tokens


def top_indices(scores: np.ndarray, top_k: int) -> list[int]:
    """
    Indices of the top_k scores, highest first; ties go to the earlier chunk
    so rankings are deterministic.
    """
    order = np.lexsort((np.arange(len(scores)), -scores))
    return [int(i) for i in order[:top_k]]


def make_results(chunks: list[dict], scores: np.ndarray, top_k: int) -> list[dict]:
    results = []
    for rank, index in enumerate(top_indices(scores, top_k), start=1):
        chunk = chunks[index]
        results.append(
            {
                "rank": rank,
                "score": float(scores[index]),
                "chunk_id": chunk["chunk_id"],
                "title": chunk["title"],
                "category": chunk["category"],
                "source": chunk["source"],
                "url": chunk["url"],
                "text": chunk["text"],
            }
        )
    return results


def print_results(results: list[dict], max_chars: int = 300) -> None:
    for r in results:
        print(f"[{r['rank']}] score={r['score']:.4f}  {r['chunk_id']}")
        print(f"    {r['title']}  ({r['category']})  {r['url']}")
        text = " ".join(r["text"].split())
        if max_chars and len(text) > max_chars:
            text = text[:max_chars] + " …"
        print(textwrap.indent(textwrap.fill(text, 100), "    "))
        print()


def add_query_arguments(parser) -> None:
    """
    Arguments shared by the retriever CLIs.
    """
    parser.add_argument("--query", required=True, help="Natural-language question.")
    parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to return.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Retrieval config JSON.")
    parser.add_argument("--max-chars", type=int, default=300, help="Characters of chunk text to print (0 = all).")
    parser.add_argument("--json", action="store_true", help="Print results as JSON.")


def output_results(results: list[dict], args) -> None:
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print_results(results, args.max_chars)
