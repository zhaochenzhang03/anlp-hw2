"""
Score saved predictions against the reference answers.

Writes <run dir>/metrics.json with overall and per-topic means of exact
match, F1, answer recall and ROUGE-L (see evaluation/metrics.py), plus
per-question scores.

    python -m scripts.score_rag outputs/rag/<run-name>/predictions.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import METRICS, score  # noqa: E402
from scripts.evaluate_retrieval import TOPIC_ORDER, question_topic  # noqa: E402


def mean_scores(rows: list[dict]) -> dict:
    entry = {"n": len(rows)}
    for name in METRICS:
        entry[name] = round(sum(r["scores"][name] for r in rows) / len(rows), 4) if rows else 0.0
    return entry


def score_run(predictions_path: str | Path, questions: list[dict]) -> dict:
    """
    Score every question in `questions`; a question without a prediction
    counts as an empty answer, so incomplete runs are not overrated.
    """
    predictions_path = Path(predictions_path)
    predictions = {}
    with predictions_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            predictions[record["id"]] = record["answer"]

    rows = []
    for q in questions:
        answer = predictions.get(q["id"], "")
        rows.append(
            {
                "id": q["id"],
                "topic": question_topic(q["question"]),
                "answer": answer,
                "missing": q["id"] not in predictions,
                "scores": score(answer, q["answers"]),
            }
        )

    summary = {
        "predictions": str(predictions_path),
        "n_questions": len(questions),
        "n_predicted": sum(not r["missing"] for r in rows),
        "overall": mean_scores(rows),
        "by_topic": {t: mean_scores([r for r in rows if r["topic"] == t]) for t in TOPIC_ORDER},
        "per_question": rows,
    }

    with (predictions_path.parent / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Score RAG predictions against reference answers.")
    parser.add_argument("predictions", nargs="+", help="predictions.jsonl file(s).")
    parser.add_argument("--questions", default="data/eval/public_evaluation_data.json")
    args = parser.parse_args()

    with open(args.questions, "r", encoding="utf-8") as f:
        questions = json.load(f)

    for path in args.predictions:
        summary = score_run(path, questions)
        overall = summary["overall"]
        print(
            f"{path}: n={summary['n_predicted']}/{summary['n_questions']}  "
            + "  ".join(f"{k}={overall[k]:.4f}" for k in METRICS)
        )


if __name__ == "__main__":
    main()
