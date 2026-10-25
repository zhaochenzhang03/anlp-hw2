"""
Answer metrics named in the handout: exact match and token F1 as in SQuAD
(Rajpurkar et al., 2016, section 6.1), answer recall, and ROUGE-L. Each is
the maximum over a question's reference answers.
"""

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    """
    SQuAD normalization: lowercase, drop punctuation and the articles
    a/an/the, collapse whitespace.
    """
    text = text.lower()
    text = "".join(c for c in text if c not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def tokens(text: str) -> list[str]:
    return normalize_answer(text).split()


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def _overlap(prediction: list[str], reference: list[str]) -> int:
    return sum((Counter(prediction) & Counter(reference)).values())


def f1(prediction: str, reference: str) -> float:
    pred, ref = tokens(prediction), tokens(reference)
    if not pred or not ref:
        return float(pred == ref)
    common = _overlap(pred, ref)
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def answer_recall(prediction: str, reference: str) -> float:
    """
    Share of the reference's tokens that appear in the prediction.
    """
    pred, ref = tokens(prediction), tokens(reference)
    if not ref:
        return float(not pred)
    return _overlap(pred, ref) / len(ref)


def _lcs(a: list[str], b: list[str]) -> int:
    previous = [0] * (len(b) + 1)
    for x in a:
        current = [0]
        for j, y in enumerate(b, start=1):
            current.append(previous[j - 1] + 1 if x == y else max(previous[j], current[j - 1]))
        previous = current
    return previous[-1]


def rouge_l(prediction: str, reference: str) -> float:
    """
    ROUGE-L F1 on normalized tokens.
    """
    pred, ref = tokens(prediction), tokens(reference)
    if not pred or not ref:
        return float(pred == ref)
    lcs = _lcs(pred, ref)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(pred), lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


METRICS = {"exact_match": exact_match, "f1": f1, "recall": answer_recall, "rouge_l": rouge_l}


def score(prediction: str, references: list[str]) -> dict:
    """
    Every metric for one prediction, each maximized over the references.
    """
    return {name: max(fn(prediction, ref) for ref in references) for name, fn in METRICS.items()}
