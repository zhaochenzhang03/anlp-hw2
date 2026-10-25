"""
Turn a predictions.jsonl file into a submission JSON and validate it.

Unseen test set format:   {"1": "Answer 1", "2": "Answer 2", ...}
Leaderboard format:       {"andrewid": "<id or nickname>", "1": "Answer 1", ...}

    python -m scripts.export_system_output \
        --predictions outputs/rag/<run>/predictions.jsonl \
        --questions data/eval/test_queries.json \
        --output system_outputs/system_output_1.json

    python -m scripts.export_system_output \
        --predictions outputs/rag/<leaderboard run>/predictions.jsonl \
        --questions data/eval/leaderboard_queries.json \
        --andrewid <id> --output outputs/leaderboard/submission.json
"""

import argparse
import json
import sys
from pathlib import Path


# Generated text that must never reach a submission.
DEBUG_MARKERS = ("Context:", "Question:", "<|im_", "</s>", "[1]")


def load_predictions(path: Path) -> dict[str, str]:
    predictions = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            predictions[record["id"]] = record["answer"]
    return predictions


def build_output(questions: list[dict], predictions: dict[str, str], andrewid: str | None) -> dict:
    output = {"andrewid": andrewid} if andrewid else {}
    for q in questions:
        output[q["id"]] = predictions[q["id"]]
    return output


def validate(output: dict, questions: list[dict], andrewid: str | None) -> list[str]:
    """
    Problems with the submission; empty if it is valid.
    """
    problems = []
    keys = [k for k in output if k != "andrewid"]
    ids = [q["id"] for q in questions]

    if andrewid and output.get("andrewid") != andrewid:
        problems.append("andrewid missing")
    if keys != ids:
        problems.append(f"IDs/order differ from the question file ({len(keys)} answers, {len(ids)} questions)")

    references = {a.strip().lower() for q in questions for a in q.get("answers", [])}
    for key in keys:
        answer = output[key]
        if not isinstance(answer, str):
            problems.append(f"{key}: answer is not a string")
        elif not answer.strip():
            problems.append(f"{key}: empty answer")
        elif any(marker in answer for marker in DEBUG_MARKERS):
            problems.append(f"{key}: answer contains prompt/debug text: {answer[:60]!r}")
        elif "\n" in answer:
            problems.append(f"{key}: answer has a line break")
    if references and "answers" in questions[0]:
        # Only meaningful when exporting a dev set: flags copying references.
        copied = sum(output[k].strip().lower() in references for k in keys)
        print(f"Note: {copied} answers equal a reference string (dev set only; expected for correct answers).")

    return problems


def main():
    parser = argparse.ArgumentParser(description="Export predictions to the submission JSON format.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--questions", required=True, help="Question file the predictions were made on.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--andrewid", default=None, help="Adds the leaderboard 'andrewid' field.")
    args = parser.parse_args()

    with open(args.questions, "r", encoding="utf-8") as f:
        questions = json.load(f)
    predictions = load_predictions(Path(args.predictions))

    missing = [q["id"] for q in questions if q["id"] not in predictions]
    if missing:
        sys.exit(f"{len(missing)} questions have no prediction (first: {missing[:10]}); finish the run first.")

    output = build_output(questions, predictions, args.andrewid)
    problems = validate(output, questions, args.andrewid)
    if problems:
        sys.exit("Invalid submission:\n  " + "\n  ".join(problems))

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(questions)} answers to {path} (validated: IDs, order, non-empty strings, no prompt text).")


if __name__ == "__main__":
    main()
