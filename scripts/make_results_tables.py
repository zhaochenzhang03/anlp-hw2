"""
Collect retrieval and end-to-end results into report-ready tables.

Reads only saved outputs (nothing is re-run):
  outputs/retrieval_eval/before_corpus_patch/summary.json   retrieval, before the patch round
  outputs/retrieval_eval/summary.json                       retrieval, current corpus
  outputs/rag/<run>/metrics.json, run_config.json           end-to-end runs
  outputs/error_analysis/<primary run>.json                 failure types

Writes outputs/results/*.csv, results.json and results.md. Runs that do not
exist yet are skipped and listed as missing.

    python -m scripts.make_results_tables
"""

import csv
import json
import sys
from pathlib import Path

from scipy.stats import binomtest, ttest_rel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_retrieval import TOPIC_ORDER  # noqa: E402

RAG_DIR = Path("outputs/rag")
OUT_DIR = Path("outputs/results")
PRIMARY = "weighted0.75_k5__Qwen3-4B-Instruct-2507"
METRIC_NAMES = ("exact_match", "f1", "recall", "rouge_l")
RETRIEVAL_METHODS = {"bm25": "BM25", "dense": "Dense", "rrf": "RRF (k=60)", "weighted_a0.75": "Weighted a=0.75 (dev-tuned)"}
K_VALUES = (1, 3, 5, 10, 20)

# (table, label, run name). The first row of each table is its baseline for
# the paired tests.
RAG_TABLES = {
    "end_to_end": [
        ("Closed-book (may abstain)", "closed_book__Qwen3-4B-Instruct-2507"),
        ("Closed-book (always answers)", "closed_book_guess__Qwen3-4B-Instruct-2507"),
        ("BM25 RAG", "bm25_k5__Qwen3-4B-Instruct-2507"),
        ("Dense RAG", "dense_k5__Qwen3-4B-Instruct-2507"),
        ("RRF hybrid RAG", "rrf_k5__Qwen3-4B-Instruct-2507"),
        ("Weighted a=0.75 hybrid RAG (dev-tuned)", PRIMARY),
    ],
    "top_k": [
        ("Weighted a=0.75, top-5", PRIMARY),
        ("Weighted a=0.75, top-10", "weighted0.75_k10__Qwen3-4B-Instruct-2507"),
    ],
    "reader": [
        ("Qwen3-4B-Instruct-2507", PRIMARY),
        ("Qwen2.5-1.5B-Instruct", "weighted0.75_k5__Qwen2.5-1.5B-Instruct"),
    ],
}


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def md_table(header: list[str], rows: list[list]) -> str:
    def cell(x):
        return f"{x:.3f}" if isinstance(x, float) else str(x)

    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(cell(x) for x in row) + " |" for row in rows]
    return "\n".join(lines)


def retrieval_tables(md: list[str], results: dict) -> None:
    before = load_json(Path("outputs/retrieval_eval/before_corpus_patch/summary.json"))
    after = load_json(Path("outputs/retrieval_eval/summary.json"))

    header = ["corpus", "method", "n_chunks"] + [f"exact_hit@{k}" for k in K_VALUES] + [f"soft_hit@{k}" for k in K_VALUES]
    rows = []
    for corpus, summary in (("before patch", before), ("after patch", after)):
        if summary is None:
            continue
        for key, label in RETRIEVAL_METHODS.items():
            entry = summary["results"][key]["All"]
            rows.append([corpus, label, summary["n_chunks"]]
                        + [entry["exact"][f"hit@{k}"] for k in K_VALUES]
                        + [entry["soft"][f"hit@{k}"] for k in K_VALUES])
    write_csv(OUT_DIR / "retrieval.csv", header, rows)
    results["retrieval"] = [dict(zip(header, r)) for r in rows]
    md += ["## 1. Retrieval (public dev, 157 questions; proxy hits, no gold chunks)", "",
           md_table(header[:3] + header[3:8], [r[:8] for r in rows]), "",
           md_table(header[:3] + header[8:], [r[:3] + r[8:] for r in rows]), ""]

    header = ["topic", "n", "in_corpus_before", "in_corpus_after"] + [f"{l} hit@5 after" for l in RETRIEVAL_METHODS.values()]
    rows = []
    if before and after:
        for topic in ["All"] + TOPIC_ORDER:
            rows.append([topic, after["answer_in_corpus"][topic]["n"],
                         before["answer_in_corpus"][topic]["in_corpus"], after["answer_in_corpus"][topic]["in_corpus"]]
                        + [after["results"][k][topic]["exact"]["hit@5"] for k in RETRIEVAL_METHODS])
    write_csv(OUT_DIR / "retrieval_by_topic.csv", header, rows)
    results["retrieval_by_topic"] = [dict(zip(header, r)) for r in rows]
    md += ["### Exact-match ceiling and hit@5 by topic", "", md_table(header, rows), ""]


def per_question(run: str) -> dict | None:
    metrics = load_json(RAG_DIR / run / "metrics.json")
    if metrics is None or metrics["n_predicted"] < metrics["n_questions"]:
        return None
    return {r["id"]: r for r in metrics["per_question"]}


def paired_tests(base: dict, other: dict) -> dict:
    ids = sorted(base)
    f1_a = [base[i]["scores"]["f1"] for i in ids]
    f1_b = [other[i]["scores"]["f1"] for i in ids]
    em_a = [base[i]["scores"]["exact_match"] == 1 for i in ids]
    em_b = [other[i]["scores"]["exact_match"] == 1 for i in ids]
    only_a = sum(a and not b for a, b in zip(em_a, em_b))
    only_b = sum(b and not a for a, b in zip(em_a, em_b))
    n = only_a + only_b
    return {
        "f1_paired_t_p": round(float(ttest_rel(f1_b, f1_a).pvalue), 4) if f1_a != f1_b else 1.0,
        "em_mcnemar_p": round(float(binomtest(min(only_a, only_b), n, 0.5).pvalue), 4) if n else 1.0,
    }


def rag_tables(md: list[str], results: dict, missing: list[str]) -> None:
    titles = {
        "end_to_end": "## 2. End-to-end RAG (public dev, 157 questions; reader Qwen3-4B-Instruct-2507, top-5)",
        "top_k": "## 3. Top-k ablation",
        "reader": "## 4. Reader ablation",
    }
    for table, runs in RAG_TABLES.items():
        header = ["system", "run", "n"] + list(METRIC_NAMES) + ["minutes", "mean_prompt_tokens", "abstained", "f1_p_vs_first", "em_p_vs_first"]
        rows = []
        base = per_question(runs[0][1])
        for label, run in runs:
            metrics = load_json(RAG_DIR / run / "metrics.json")
            rows_q = per_question(run)
            if rows_q is None:
                missing.append(run)
                continue
            info = load_json(RAG_DIR / run / "run_config.json") or {"sessions": []}
            minutes = round(sum(s["seconds"] for s in info["sessions"]) / 60, 1)
            with (RAG_DIR / run / "predictions.jsonl").open() as f:
                preds = [json.loads(line) for line in f]
            tokens = round(sum(p["prompt_tokens"] for p in preds) / len(preds))
            abstained = sum(p["answer"].lower().startswith("i don't know") for p in preds)
            tests = paired_tests(base, rows_q) if base and run != runs[0][1] else {"f1_paired_t_p": "", "em_mcnemar_p": ""}
            rows.append([label, run, metrics["overall"]["n"]] + [metrics["overall"][m] for m in METRIC_NAMES]
                        + [minutes, tokens, abstained, tests["f1_paired_t_p"], tests["em_mcnemar_p"]])
        write_csv(OUT_DIR / f"rag_{table}.csv", header, rows)
        results[f"rag_{table}"] = [dict(zip(header, r)) for r in rows]
        md += [titles[table], "", md_table([h for h in header if h != "run"], [[c for j, c in enumerate(r) if j != 1] for r in rows]), ""]

    # Topic-level F1 for every end-to-end system.
    header = ["topic", "n"] + [label for label, _ in RAG_TABLES["end_to_end"]]
    rows = []
    loaded = {run: load_json(RAG_DIR / run / "metrics.json") for _, run in RAG_TABLES["end_to_end"]}
    first = next((m for m in loaded.values() if m), None)
    if first:
        for topic in TOPIC_ORDER:
            rows.append([topic, first["by_topic"][topic]["n"]]
                        + [loaded[run]["by_topic"][topic]["f1"] if loaded[run] else "" for _, run in RAG_TABLES["end_to_end"]])
    write_csv(OUT_DIR / "rag_by_topic_f1.csv", header, rows)
    results["rag_by_topic_f1"] = [dict(zip(header, r)) for r in rows]
    md += ["## 5. Topic-level F1 (end-to-end)", "", md_table(header, rows), ""]


def error_table(md: list[str], results: dict, missing: list[str]) -> None:
    analysis = load_json(Path("outputs/error_analysis") / f"{PRIMARY}.json")
    if analysis is None:
        missing.append(f"error_analysis/{PRIMARY}")
        return
    header = ["failure type", "all failures", "manually inspected"]
    rows = [[t, n, analysis["counts_manually_inspected"].get(t, 0)] for t, n in analysis["counts"].items()]
    rows.append(["total", analysis["n_failures"], analysis["n_manually_inspected"]])
    write_csv(OUT_DIR / "error_types.csv", header, rows)
    results["error_types"] = {"run": PRIMARY, "threshold": analysis["failure_threshold"], "rows": [dict(zip(header, r)) for r in rows]}
    md += [f"## 6. Error types ({PRIMARY}, failures = {analysis['failure_threshold']})", "", md_table(header, rows), ""]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md = ["# Results tables", "", "Generated by `python -m scripts.make_results_tables` from saved outputs.", ""]
    results, missing = {}, []

    retrieval_tables(md, results)
    rag_tables(md, results, missing)
    error_table(md, results, missing)

    missing = list(dict.fromkeys(missing))
    if missing:
        md += ["## Not available (not run or incomplete)", ""] + [f"- {m}" for m in missing] + [""]
    results["missing"] = missing

    (OUT_DIR / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    (OUT_DIR / "results.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
