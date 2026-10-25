"""
Fast checks that need no model download: metrics, prompt building, answer
cleaning, resume parsing and submission validation.

    python -m unittest discover tests
"""

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.metrics import score
from generation.prompts import build_messages
from scripts.export_system_output import build_output, validate
from scripts.run_rag_eval import load_done


class MetricsTest(unittest.TestCase):
    def test_exact_and_f1(self):
        s = score("The Tepper School", ["Tepper School"])
        self.assertEqual(s["exact_match"], 1.0)
        self.assertEqual(s["f1"], 1.0)

    def test_max_over_references(self):
        s = score("December 10", ["December 10, 2026", "December 10"])
        self.assertEqual(s["exact_match"], 1.0)

    def test_partial_overlap(self):
        s = score("Tepper School of Business", ["Carnegie Mellon's business school is the Tepper School of Business."])
        self.assertEqual(s["exact_match"], 0.0)
        self.assertGreater(s["f1"], 0.5)
        self.assertGreater(s["rouge_l"], 0.5)
        self.assertLess(s["recall"], 1.0)

    def test_empty_prediction(self):
        self.assertEqual(score("", ["x"])["f1"], 0.0)


class PromptTest(unittest.TestCase):
    def test_rag_prompt_contains_numbered_context(self):
        chunks = [{"chunk_id": "a#0", "title": "T", "url": "u", "text": "Fact one."}]
        messages = build_messages("Q?", chunks)
        self.assertIn("[1] T (u)\nFact one.", messages[1]["content"])
        self.assertTrue(messages[1]["content"].endswith("Question: Q?\nAnswer:"))

    def test_closed_book_has_no_context(self):
        messages = build_messages("Q?", None)
        self.assertNotIn("Context:", messages[1]["content"])


class ReaderHelpersTest(unittest.TestCase):
    def test_clean_answer(self):
        from generation.reader import clean_answer

        self.assertEqual(clean_answer("Answer: William Pitt\nextra"), "William Pitt")
        self.assertEqual(clean_answer("\n  1787  \n"), "1787")


class ResumeTest(unittest.TestCase):
    def test_truncated_last_line_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.jsonl"
            path.write_text(json.dumps({"id": "1", "answer": "a"}) + "\n" + '{"id": "2", "ans')
            self.assertEqual(list(load_done(path)), ["1"])


class ExportTest(unittest.TestCase):
    questions = [{"id": "1", "question": "a"}, {"id": "2", "question": "b"}]

    def test_valid_output(self):
        out = build_output(self.questions, {"1": "x", "2": "y"}, None)
        self.assertEqual(list(out), ["1", "2"])
        self.assertEqual(validate(out, self.questions, None), [])

    def test_leaderboard_field_first(self):
        out = build_output(self.questions, {"1": "x", "2": "y"}, "nick")
        self.assertEqual(list(out)[0], "andrewid")
        self.assertEqual(validate(out, self.questions, "nick"), [])

    def test_rejects_empty_and_prompt_text(self):
        out = build_output(self.questions, {"1": "", "2": "Question: b"}, None)
        self.assertEqual(len(validate(out, self.questions, None)), 2)


if __name__ == "__main__":
    unittest.main()
