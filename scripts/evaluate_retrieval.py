import argparse
import csv
import json
import re
import sys
from itertools import combinations
from pathlib import Path

from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.bm25 import BM25Retriever  # noqa: E402
from retrieval.common import (  # noqa: E402
    canonical_months,
    load_chunks,
    load_config,
    normalize_text,
    tokenize,
    top_indices,
)
from retrieval.dense import DenseRetriever  # noqa: E402
from retrieval.hybrid import fuse  # noqa: E402


# Question topics, checked in order; the first match wins.
TOPICS = [
    ("Carnegie Museum of Art", r"Carnegie Museum of Art|CMoA"),
    (
        "Sports",
        r"Steelers|Penguins|Pirates|Pitt\b.*(basketball|volleyball|football|soccer)|"
        r"Duquesne|Chatham|Robert Morris|Pitt women|Pitt men",
    ),
    ("Taxes/budget", r"\btax|budget|LST|millage|police officers"),
    ("CMU", r"Carnegie Mellon|CMU"),
    (
        "Events/music/comedy",
        r"perform|comedian|event|concert|open mic|Improv|show|festival|production|"
        r"Opera|Symphony|Music Hall|auditions",
    ),
    (
        "Restaurants/food",
        r"restaurant|\bbar\b|food|dining|cuisine|bakery|coffee|chocolate|pizza|"
        r"steakhouse|whiskey|Fet Fisk|täkō|Pusadee|macaron|Market",
    ),
]
DEFAULT_TOPIC = "Pittsburgh local/history"

# Answers found in at most this many chunks count as specific: an exact hit
# on them is unlikely to come from an unrelated chunk.
SPECIFIC_MAX_CHUNKS = 10
SPECIFIC_GROUP = f"Specific answers (in 1-{SPECIFIC_MAX_CHUNKS} chunks)"
TOPIC_ORDER = [name for name, _ in TOPICS[:1]] + ["CMU", DEFAULT_TOPIC, "Events/music/comedy", "Sports", "Taxes/budget", "Restaurants/food"]


def question_topic(question: str) -> str:
    for name, pattern in TOPICS:
        if re.search(pattern, question, re.IGNORECASE):
            return name
    return DEFAULT_TOPIC


def normalize_answer(text: str) -> str:
    """
    Lowercase, strip accents, shorten month names, split digits from letters
    ("8:00pm" -> "8 00 pm"), turn punctuation into spaces, collapse spaces.
    """
    text = canonical_months(normalize_text(text))
    text = re.sub(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)", " ", text)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def answer_tokens(text: str) -> set[str]:
    """
    Content tokens of an answer; all tokens if it has only stopwords.
    """
    tokens = normalize_answer(text).split()
    return set(tokenize(" ".join(tokens))) or set(tokens)


class AnswerMatcher:
    """
    Decides whether a chunk contains a reference answer.

    exact: the normalized answer occurs in the normalized chunk on token
           boundaries (so "12" does not match "120").
    soft:  at least `threshold` of the answer's content tokens occur in the
           chunk. Every exact match is also a soft match.
    """

    def __init__(self, chunks: list[dict], threshold: float):
        self.padded = [f" {normalize_answer(c['text'])} " for c in chunks]
        self.token_sets = [set(normalize_answer(c["text"]).split()) for c in chunks]
        self.threshold = threshold

    def exact(self, answers: list[str], index: int) -> bool:
        text = self.padded[index]
        return any(a and f" {a} " in text for a in answers)

    def soft(self, answer_sets: list[set[str]], index: int) -> bool:
        tokens = self.token_sets[index]
        return any(s and len(s & tokens) / len(s) >= self.threshold for s in answer_sets)


def first_rank(ranked: list[int], matches) -> int | None:
    for rank, index in enumerate(ranked, start=1):
        if matches(index):
            return rank
    return None


def method_specs(config: dict) -> list[tuple[str, str, float]]:
    """
    (name, method, alpha) for every evaluated retriever.
    """
    specs = [("bm25", "bm25", 0.0), ("dense", "dense", 1.0), ("rrf", "rrf", 0.0)]
    for alpha in config["hybrid"]["alphas"]:
        specs.append((f"weighted_a{alpha}", "weighted", alpha))
    return specs


def hit_rate(ranks: list[int | None], k: int) -> float:
    return sum(r is not None and r <= k for r in ranks) / len(ranks) if ranks else 0.0


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    n = only_a + only_b
    p = binomtest(min(only_a, only_b), n, 0.5).pvalue if n else 1.0
    return {"only_first": only_a, "only_second": only_b, "p_value": round(float(p), 6)}


def main():
    parser = argparse.ArgumentParser(description="Retrieval-only evaluation on the public dev set.")
    parser.add_argument("--config", default="configs/retrieval_config.json")
    parser.add_argument("--output-dir", default=None, help="Defaults to evaluation.output_dir in the config.")
    args = parser.parse_args()

    config = load_config(args.config)
    settings = config["evaluation"]
    k_values = settings["k_values"]
    max_k = max(k_values)
    output_dir = Path(args.output_dir or settings["output_dir"])

    chunks = load_chunks(config["chunks_path"])
    bm25 = BM25Retriever.from_config(config, chunks)
    dense = DenseRetriever.from_config(config, chunks)
    hybrid = config["hybrid"]

    with open(settings["eval_path"], "r", encoding="utf-8") as f:
        questions = json.load(f)

    matcher = AnswerMatcher(chunks, settings["soft_threshold"])
    specs = method_specs(config)

    per_question = []
    for q in questions:
        answers = [normalize_answer(a) for a in q["answers"]]
        answer_sets = [answer_tokens(a) for a in q["answers"]]

        sparse_scores = bm25.scores(q["question"])
        dense_scores = dense.scores(q["question"])

        record = {
            "id": q["id"],
            "question": q["question"],
            "answers": q["answers"],
            "topic": question_topic(q["question"]),
        }
        # How many chunks contain the answer anywhere. 0 means no retriever
        # can get an exact hit; a large number means the answer is generic
        # (e.g. "Gift") and an exact hit may come from an unrelated chunk.
        n_answer_chunks = sum(matcher.exact(answers, i) for i in range(len(chunks)))
        record.update({
            "answer_in_corpus": n_answer_chunks > 0,
            "n_answer_chunks": n_answer_chunks,
            "methods": {},
        })

        for name, method, alpha in specs:
            scores = fuse(
                method, sparse_scores, dense_scores, alpha,
                rrf_k=hybrid["rrf_k"], rrf_window=hybrid["rrf_window"],
            )
            ranked = top_indices(scores, max_k)
            record["methods"][name] = {
                "exact_rank": first_rank(ranked, lambda i: matcher.exact(answers, i)),
                "soft_rank": first_rank(ranked, lambda i: matcher.soft(answer_sets, i)),
                "top": [[chunks[i]["chunk_id"], round(float(scores[i]), 6)] for i in ranked],
            }

        per_question.append(record)

    # Aggregate.
    groups = {"All": per_question}
    for topic in TOPIC_ORDER:
        groups[topic] = [r for r in per_question if r["topic"] == topic]
    groups[SPECIFIC_GROUP] = [r for r in per_question if 1 <= r["n_answer_chunks"] <= SPECIFIC_MAX_CHUNKS]

    summary = {
        "n_questions": len(per_question),
        "n_chunks": len(chunks),
        "config": config,
        "results": {},
        "answer_in_corpus": {
            group: {"n": len(rows), "in_corpus": sum(r["answer_in_corpus"] for r in rows)}
            for group, rows in groups.items()
        },
    }
    csv_rows = []

    for name, _, _ in specs:
        summary["results"][name] = {}
        for group, rows in groups.items():
            entry = {"n": len(rows)}
            for metric in ("exact", "soft"):
                ranks = [r["methods"][name][f"{metric}_rank"] for r in rows]
                entry[metric] = {f"hit@{k}": round(hit_rate(ranks, k), 4) for k in k_values}
                for k in k_values:
                    hits = sum(r is not None and r <= k for r in ranks)
                    csv_rows.append([name, group, metric, k, hits, len(rows), round(hit_rate(ranks, k), 4)])
            summary["results"][name][group] = entry

    hits_at_5 = {
        name: [r["methods"][name]["exact_rank"] is not None and r["methods"][name]["exact_rank"] <= 5 for r in per_question]
        for name, _, _ in specs
    }
    summary["mcnemar_exact_hit@5"] = {
        f"{a} vs {b}": mcnemar(hits_at_5[a], hits_at_5[b])
        for a, b in combinations([name for name, _, _ in specs], 2)
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "topic", "metric", "k", "hits", "n", "rate"])
        writer.writerows(csv_rows)

    with (output_dir / "per_question.jsonl").open("w", encoding="utf-8") as f:
        for record in per_question:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Console tables.
    in_corpus = summary["answer_in_corpus"]["All"]["in_corpus"]
    print(f"Questions: {len(per_question)}; answer found verbatim somewhere in the corpus: {in_corpus}")
    for metric in ("exact", "soft"):
        print(f"\n{metric} match, all questions")
        print("method".ljust(16) + "".join(f"hit@{k}".rjust(8) for k in k_values))
        for name, _, _ in specs:
            entry = summary["results"][name]["All"][metric]
            print(name.ljust(16) + "".join(f"{entry[f'hit@{k}']:8.3f}" for k in k_values))

    print("\nexact hit@5 by topic")
    print("group".ljust(26) + "n".rjust(4) + "corpus".rjust(8) + "".join(name[:10].rjust(11) for name, _, _ in specs))
    for group in groups:
        n = summary["answer_in_corpus"][group]["n"]
        ceiling = summary["answer_in_corpus"][group]["in_corpus"]
        cells = "".join(f"{summary['results'][name][group]['exact']['hit@5']:11.3f}" for name, _, _ in specs)
        print(group[:25].ljust(26) + f"{n:4d}{ceiling:8d}" + cells)

    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
