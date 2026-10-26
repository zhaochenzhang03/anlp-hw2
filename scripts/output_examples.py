"""
Side-by-side output examples from saved runs, for the report.

For each pair of systems, lists questions where one system's answer is
correct (token F1 >= 0.5) and the other's is not, plus questions both get
wrong. Reads only saved predictions; writes outputs/results/examples.md.

    python -m scripts.output_examples
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import score  # noqa: E402

RAG_DIR = Path("outputs/rag")
OUT = Path("outputs/results/examples.md")
PRIMARY = "weighted0.75_k5__Qwen3-4B-Instruct-2507"
PAIRS = [
    ("Closed-book", "closed_book_guess__Qwen3-4B-Instruct-2507", "Weighted hybrid RAG", PRIMARY),
    ("BM25 RAG", "bm25_k5__Qwen3-4B-Instruct-2507", "Dense RAG", "dense_k5__Qwen3-4B-Instruct-2507"),
    ("Dense RAG", "dense_k5__Qwen3-4B-Instruct-2507", "Weighted hybrid RAG", PRIMARY),
]
PER_BUCKET = 4
CORRECT_F1 = 0.5


def load(run: str) -> dict[str, dict] | None:
    path = RAG_DIR / run / "predictions.jsonl"
    if not path.exists():
        return None
    with path.open() as f:
        records = {r["id"]: r for r in map(json.loads, f)}
    for r in records.values():
        r["f1"] = score(r["answer"], r["references"])["f1"]
    return records


def row(q: dict, a: dict, b: dict) -> str:
    def cell(text: str) -> str:
        return text.replace("|", "/").replace("\n", " ")[:90]

    return (f"| {q['id']} | {q['topic']} | {cell(q['question'])} | {cell(q['references'][0])} "
            f"| {cell(a['answer'])} ({a['f1']:.2f}) | {cell(b['answer'])} ({b['f1']:.2f}) |")


def main():
    md = ["# Output examples (public dev)", "",
          f"Correct = token F1 >= {CORRECT_F1}. Picked by question ID order within each bucket; F1 in parentheses.", ""]
    for name_a, run_a, name_b, run_b in PAIRS:
        a, b = load(run_a), load(run_b)
        if a is None or b is None or set(a) != set(b):
            md += [f"## {name_a} vs {name_b}", "", "Not available (a run is missing or incomplete).", ""]
            continue
        ids = sorted(a, key=int)
        buckets = {
            f"{name_b} right, {name_a} wrong": [i for i in ids if b[i]["f1"] >= CORRECT_F1 > a[i]["f1"]],
            f"{name_a} right, {name_b} wrong": [i for i in ids if a[i]["f1"] >= CORRECT_F1 > b[i]["f1"]],
            "both wrong": [i for i in ids if a[i]["f1"] < CORRECT_F1 and b[i]["f1"] < CORRECT_F1],
        }
        md += [f"## {name_a} vs {name_b}", ""]
        for title, chosen in buckets.items():
            md += [f"### {title} ({len(chosen)} questions)", "",
                   f"| id | topic | question | reference | {name_a} | {name_b} |", "|---|---|---|---|---|---|"]
            md += [row(a[i], a[i], b[i]) for i in chosen[:PER_BUCKET]] + [""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
