"""
End-to-end RAG (or closed-book) inference over a question file.

Each prediction is appended to a JSONL file as soon as it is generated, so
an interrupted run can be resumed: completed question IDs are skipped.
Questions are processed in file order. If the question file has reference
answers, the run is scored at the end (see scripts/score_rag.py).

Examples:
    python -m scripts.run_rag_eval --method weighted --alpha 0.75 --top-k 5
    python -m scripts.run_rag_eval --method closed_book
    python -m scripts.run_rag_eval --method rrf --start 0 --end 20
    python -m scripts.run_rag_eval --questions data/eval/leaderboard_queries.json --run-name leaderboard_weighted
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generation.reader import Reader, load_generation_config  # noqa: E402
from retrieval.common import file_sha1, load_config  # noqa: E402
from scripts.evaluate_retrieval import question_topic  # noqa: E402
from scripts.score_rag import score_run  # noqa: E402

METHODS = ("closed_book", "closed_book_guess", "bm25", "dense", "rrf", "weighted")

# Settings that must match for a resumed run to append to an existing file.
RESUME_KEYS = ("method", "alpha", "top_k", "model_name", "prompt_version", "max_new_tokens", "chunks_sha1")


def default_run_name(args, model_name: str) -> str:
    model = model_name.rsplit("/", 1)[-1]
    if args.method.startswith("closed_book"):
        return f"{args.method}__{model}"
    method = f"weighted{args.alpha}" if args.method == "weighted" else args.method
    return f"{method}_k{args.top_k}__{model}"


def select_questions(questions: list[dict], args) -> list[dict]:
    if args.ids:
        wanted = set(args.ids.split(","))
        questions = [q for q in questions if q["id"] in wanted]
    return questions[args.start:args.end]


def load_done(path: Path) -> dict[str, dict]:
    """
    Completed predictions by question ID; a truncated last line (from a
    crash mid-write) is ignored.
    """
    done = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done[record["id"]] = record
    return done


def append_record(f, record: dict) -> None:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
    f.flush()
    os.fsync(f.fileno())


def main():
    parser = argparse.ArgumentParser(description="Run RAG or closed-book inference and save predictions.")
    parser.add_argument("--questions", default="data/eval/public_evaluation_data.json")
    parser.add_argument("--method", choices=METHODS, default=None, help="Defaults to the generation config.")
    parser.add_argument("--alpha", type=float, default=None, help="Dense weight for --method weighted.")
    parser.add_argument("--top-k", type=int, default=None, help="Chunks given to the reader.")
    parser.add_argument("--model", default=None, help="Hugging Face model ID (overrides the config).")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1, help="Questions per generate() call.")
    parser.add_argument("--device", default=None, help="cuda, mps, cpu or auto.")
    parser.add_argument("--gen-config", default="configs/generation_config.json")
    parser.add_argument("--retrieval-config", default="configs/retrieval_config.json")
    parser.add_argument("--output-dir", default="outputs/rag", help="Runs are saved in <output-dir>/<run-name>/.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--start", type=int, default=0, help="First question index (after --ids filtering).")
    parser.add_argument("--end", type=int, default=None, help="Stop before this index.")
    parser.add_argument("--ids", default=None, help="Comma-separated question IDs to run.")
    parser.add_argument("--overwrite", action="store_true", help="Discard existing predictions instead of resuming.")
    args = parser.parse_args()

    gen_config = load_generation_config(args.gen_config)
    if args.model:
        gen_config["model_name"] = args.model
        # Pinned commit for models listed in the config; latest otherwise.
        gen_config["revision"] = gen_config.get("known_revisions", {}).get(args.model)
    if args.max_new_tokens:
        gen_config["max_new_tokens"] = args.max_new_tokens
    if args.device:
        gen_config["device"] = args.device

    retrieval_defaults = gen_config["retrieval"]
    args.method = args.method or retrieval_defaults["method"]
    args.alpha = args.alpha if args.alpha is not None else retrieval_defaults["alpha"]
    args.top_k = args.top_k or retrieval_defaults["top_k"]
    if args.method != "weighted":
        args.alpha = None
    if args.method.startswith("closed_book"):
        args.top_k = 0
    gen_config["closed_book_always_answer"] = args.method == "closed_book_guess"

    run_name = args.run_name or default_run_name(args, gen_config["model_name"])
    run_dir = Path(args.output_dir) / run_name
    predictions_path = run_dir / "predictions.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(args.questions, "r", encoding="utf-8") as f:
        all_questions = json.load(f)
    questions = select_questions(all_questions, args)

    retrieval_config = load_config(args.retrieval_config)
    chunks_sha1 = file_sha1(retrieval_config["chunks_path"])

    if args.overwrite and predictions_path.exists():
        predictions_path.unlink()
    done = load_done(predictions_path)

    settings = {
        "method": args.method,
        "alpha": args.alpha,
        "top_k": args.top_k,
        "model_name": gen_config["model_name"],
        "prompt_version": None,
        "max_new_tokens": gen_config["max_new_tokens"],
        "chunks_sha1": None if args.method.startswith("closed_book") else chunks_sha1,
    }

    todo = [q for q in questions if q["id"] not in done]
    print(f"Run: {run_dir}\nQuestions selected: {len(questions)}; already done: {len(questions) - len(todo)}; to run: {len(todo)}")

    retriever = None
    if todo and not args.method.startswith("closed_book"):
        from retrieval.hybrid import HybridRetriever

        retriever = HybridRetriever.from_config(retrieval_config)

    reader = Reader(gen_config) if todo else None
    if reader:
        settings["prompt_version"] = reader.metadata()["prompt_version"]
        # Refuse to mix predictions from different settings in one file.
        for record in done.values():
            mismatched = [k for k in RESUME_KEYS if record["settings"].get(k) != settings[k]]
            if mismatched:
                sys.exit(f"Existing predictions in {predictions_path} differ in {mismatched}; use --overwrite or another --run-name.")
            break

    start_time = time.perf_counter()
    with predictions_path.open("a", encoding="utf-8") as f:
        for offset in range(0, len(todo), args.batch_size):
            batch = todo[offset:offset + args.batch_size]

            retrieved = []
            for q in batch:
                if retriever is None:
                    retrieved.append(None)
                else:
                    alpha = args.alpha if args.alpha is not None else 0.5
                    retrieved.append(retriever.search(q["question"], args.top_k, args.method, alpha))

            outputs = reader.answer_batch([q["question"] for q in batch], retrieved)

            for q, chunks, out in zip(batch, retrieved, outputs):
                record = {
                    "id": q["id"],
                    "question": q["question"],
                    "topic": question_topic(q["question"]),
                    "answer": out["answer"],
                    "references": q.get("answers"),
                    "raw_output": out["raw_output"],
                    "retrieved": [
                        {"rank": c["rank"], "chunk_id": c["chunk_id"], "score": round(c["score"], 6)}
                        for c in chunks
                    ] if chunks is not None else [],
                    "context_chunk_ids": out["context_chunk_ids"],
                    "prompt_tokens": out["prompt_tokens"],
                    "generation_seconds": out["generation_seconds"],
                    "settings": settings,
                    "reader": reader.metadata(),
                }
                append_record(f, record)
                print(f"[{q['id']}] {out['answer'][:80]!r}  ({out['generation_seconds']}s, {out['prompt_tokens']} prompt tokens)")

    elapsed = time.perf_counter() - start_time

    run_info_path = run_dir / "run_config.json"
    run_info = json.loads(run_info_path.read_text()) if run_info_path.exists() else {"sessions": []}
    run_info.update(
        {
            "run_name": run_name,
            "questions_file": args.questions,
            "settings": settings,
            "generation_config": gen_config,
            "retrieval_config": None if args.method.startswith("closed_book") else retrieval_config,
            "reader": reader.metadata() if reader else run_info.get("reader"),
        }
    )
    run_info["sessions"].append(
        {
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "start": args.start,
            "end": args.end,
            "ids": args.ids,
            "questions_run": len(todo),
            "seconds": round(elapsed, 1),
        }
    )
    run_info_path.write_text(json.dumps(run_info, indent=2, ensure_ascii=False))

    total = len(load_done(predictions_path))
    print(f"\nSaved {len(todo)} new predictions in {elapsed:.0f}s; {total} total in {predictions_path}")

    if all("answers" in q for q in all_questions):
        summary = score_run(predictions_path, all_questions)
        print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
