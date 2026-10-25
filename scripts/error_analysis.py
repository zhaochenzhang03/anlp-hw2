"""
End-to-end error analysis for one RAG run.

For every question whose answer scores F1 < 0.5, suggest a failure type
from the evidence (is the answer in the corpus, in the retrieved top 20,
in the reader's context?), then apply manual labels from a JSON file
({"<id>": {"type": ..., "note": ...}}), which take precedence. Writes
outputs/error_analysis/<run-name>.{json,csv} and prints the counts.

    python -m scripts.error_analysis outputs/rag/<run-name>/predictions.jsonl \
        --labels outputs/error_analysis/manual_labels.json
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import score  # noqa: E402
from retrieval.common import load_chunks, load_config  # noqa: E402
from scripts.evaluate_retrieval import AnswerMatcher, answer_tokens, normalize_answer  # noqa: E402

TYPES = {
    "corpus": "corpus failure: the correct fact is absent from the corpus",
    "retrieval": "retrieval failure: the fact is in the corpus but not retrieved into the context",
    "context_noise": "context-selection/noise failure: the right chunk is in the context but buried among distractors",
    "generation": "generation failure: the evidence is in the context but the model answers incorrectly",
    "format_mismatch": "formatting/evaluation mismatch: substantively correct, penalized for format or paraphrase",
    "ambiguity_stale": "ambiguity/stale source: conflicting or outdated values in the sources",
}

FAIL_F1 = 0.5


def suggest_type(record: dict, scores: dict, matcher: AnswerMatcher, index_of: dict) -> tuple[str, dict]:
    """
    Heuristic failure type and the evidence behind it.
    """
    answers = [normalize_answer(a) for a in record["references"]]
    answer_sets = [answer_tokens(a) for a in record["references"]]

    def contains(chunk_id: str) -> bool:
        i = index_of[chunk_id]
        return matcher.exact(answers, i) or matcher.soft(answer_sets, i)

    context = record["context_chunk_ids"]
    retrieved = [r["chunk_id"] for r in record["retrieved"]]
    in_context = [rank for rank, cid in enumerate(context, start=1) if contains(cid)]
    in_corpus = any(matcher.exact(answers, i) or matcher.soft(answer_sets, i) for i in range(len(matcher.padded)))

    evidence = {
        "answer_in_corpus": in_corpus,
        "answer_ranks_in_context": in_context,
        "answer_in_retrieved": any(contains(cid) for cid in retrieved),
    }

    if scores["recall"] >= 0.5 or (scores["f1"] > 0 and record["answer"] and scores["recall"] >= 0.34):
        return "format_mismatch", evidence
    if not in_corpus:
        return "corpus", evidence
    if not in_context:
        return "retrieval", evidence
    if in_context[0] > 1:
        return "context_noise", evidence
    return "generation", evidence


def main():
    parser = argparse.ArgumentParser(description="Classify end-to-end failures of one RAG run.")
    parser.add_argument("predictions")
    parser.add_argument("--labels", default="outputs/error_analysis/manual_labels.json")
    parser.add_argument("--retrieval-config", default="configs/retrieval_config.json")
    parser.add_argument("--output-dir", default="outputs/error_analysis")
    args = parser.parse_args()

    config = load_config(args.retrieval_config)
    chunks = load_chunks(config["chunks_path"])
    index_of = {c["chunk_id"]: i for i, c in enumerate(chunks)}
    matcher = AnswerMatcher(chunks, config["evaluation"]["soft_threshold"])

    labels_path = Path(args.labels)
    labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}

    predictions_path = Path(args.predictions)
    run_name = predictions_path.parent.name
    with predictions_path.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    rows = []
    for record in records:
        scores = score(record["answer"], record["references"])
        if scores["f1"] >= FAIL_F1:
            continue
        suggested, evidence = suggest_type(record, scores, matcher, index_of)
        manual = labels.get(record["id"], {})
        rows.append(
            {
                "id": record["id"],
                "topic": record["topic"],
                "question": record["question"],
                "answer": record["answer"],
                "references": record["references"],
                "f1": round(scores["f1"], 3),
                "suggested_type": suggested,
                "type": manual.get("type", suggested),
                "manually_inspected": bool(manual),
                "note": manual.get("note", ""),
                **evidence,
                "context_chunk_ids": record["context_chunk_ids"],
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter(r["type"] for r in rows)
    summary = {
        "run": run_name,
        "n_questions": len(records),
        "n_failures": len(rows),
        "failure_threshold": f"F1 < {FAIL_F1}",
        "n_manually_inspected": sum(r["manually_inspected"] for r in rows),
        "type_definitions": TYPES,
        "counts": {t: counts.get(t, 0) for t in TYPES},
        "counts_manually_inspected": dict(Counter(r["type"] for r in rows if r["manually_inspected"])),
        "failures": rows,
    }
    (output_dir / f"{run_name}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    with (output_dir / f"{run_name}.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "topic", "type", "suggested_type", "manually_inspected", "f1", "question", "answer", "references", "note"])
        for r in rows:
            writer.writerow([r["id"], r["topic"], r["type"], r["suggested_type"], r["manually_inspected"], r["f1"],
                             r["question"], r["answer"], " || ".join(r["references"]), r["note"]])

    print(f"{run_name}: {len(rows)} failures (F1 < {FAIL_F1}) of {len(records)}; manually inspected: {summary['n_manually_inspected']}")
    for t in TYPES:
        print(f"  {t:16} {counts.get(t, 0)}")


if __name__ == "__main__":
    main()
