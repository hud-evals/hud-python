"""Text normalization, answer comparisons, and token-level metrics.

Each comparison returns a ``float`` in ``[0.0, 1.0]`` for use as a
``SubScore.value`` or yielded directly from a task.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCTUATION_RE = re.compile(r"[^\w\s]")


def normalize(text: str | Any) -> str:
    """Normalize text for comparison: lowercase, strip punctuation and articles.

    Useful as a building block before comparing agent answers to reference
    strings. Removes noise that shouldn't affect whether an answer is correct.

    Example::

        normalize("  The Answer is: 42! ")  # "answer is 42"
    """
    s = str(text) if not isinstance(text, str) else text
    s = s.lower()
    s = _PUNCTUATION_RE.sub(" ", s)
    s = _ARTICLES_RE.sub(" ", s)
    s = _WHITESPACE_RE.sub(" ", s)
    return s.strip()


def exact_match(
    answer: str | Any,
    expected: str,
    *,
    normalize_text: bool = True,
) -> float:
    """1.0 if answer matches expected after normalization, 0.0 otherwise."""
    if normalize_text:
        return 1.0 if normalize(answer) == normalize(expected) else 0.0

    a = str(answer).strip().lower() if not isinstance(answer, str) else answer.strip().lower()
    return 1.0 if a == expected.strip().lower() else 0.0


def contains(
    answer: str | Any,
    substring: str,
    *,
    case_sensitive: bool = False,
) -> float:
    """1.0 if answer contains substring, 0.0 otherwise."""
    a = str(answer) if not isinstance(answer, str) else answer
    s = substring

    if not case_sensitive:
        a = a.lower()
        s = s.lower()

    return 1.0 if s in a else 0.0


def contains_any(
    answer: str | Any,
    substrings: list[str],
    *,
    case_sensitive: bool = False,
) -> float:
    """1.0 if answer contains at least one of the substrings, 0.0 otherwise."""
    a = str(answer) if not isinstance(answer, str) else answer

    if not case_sensitive:
        a = a.lower()
        substrings = [s.lower() for s in substrings]

    return 1.0 if any(s in a for s in substrings) else 0.0


def contains_all(
    answer: str | Any,
    substrings: list[str],
    *,
    case_sensitive: bool = False,
) -> float:
    """1.0 if answer contains all substrings, 0.0 otherwise."""
    a = str(answer) if not isinstance(answer, str) else answer

    if not case_sensitive:
        a = a.lower()
        substrings = [s.lower() for s in substrings]

    return 1.0 if all(s in a for s in substrings) else 0.0


def numeric_match(
    answer: str | Any,
    expected: float,
    *,
    tolerance: float = 0.0,
) -> float:
    """1.0 if the first number in the answer matches expected (within tolerance)."""
    a = str(answer) if not isinstance(answer, str) else answer
    match = re.search(r"-?\d+\.?\d*", a)
    if not match:
        return 0.0

    try:
        found = float(match.group())
    except ValueError:
        return 0.0

    return 1.0 if abs(found - expected) <= tolerance else 0.0


def _tokenize(text: str) -> list[str]:
    """Tokenize normalized text into words."""
    return normalize(text).split()


def f1_score(
    answer: str | Any,
    reference: str,
) -> float:
    """Token-level F1 between answer and reference.

    Normalizes both texts, tokenizes into words, then computes
    precision, recall, and their harmonic mean.

    Example::

        f1_score("The capital is Paris, France", "Paris")  # 0.4
        f1_score("Paris", "Paris")  # 1.0
    """
    pred_tokens = _tokenize(str(answer))
    ref_tokens = _tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)


__all__ = [
    "contains",
    "contains_all",
    "contains_any",
    "exact_match",
    "f1_score",
    "normalize",
    "numeric_match",
]
