"""Reusable grading helpers for scenario evaluate phases.

These helpers reduce boilerplate in the second yield of scenarios.
Each returns a float (0.0-1.0) or a ScenarioResult with subscores
for more detailed evaluation breakdowns.

Usage in a scenario::

    from hud.native.graders import exact_match, contains_any, checklist


    @env.scenario("lookup")
    async def lookup(city: str):
        answer = yield f"What country is {city} in?"
        yield exact_match(answer, "France")


    @env.scenario("find-brands")
    async def find_brands():
        answer = yield "Name three Japanese car brands"
        yield contains_any(answer, ["toyota", "honda", "nissan", "mazda", "subaru"])


    @env.scenario("checkout")
    async def checkout(product: str):
        answer = yield f"Add {product} to cart and checkout"
        yield await checklist(
            ("in_cart", product_in_cart(product)),
            ("ordered", order_completed(product)),
            weights=[0.3, 0.7],
        )
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Awaitable
from typing import Any

from hud.tools.types import ScenarioResult, SubScore

# =============================================================================
# Text normalization
# =============================================================================

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


# =============================================================================
# Simple comparisons
# =============================================================================


def exact_match(
    answer: str | Any,
    expected: str,
    *,
    normalize_text: bool = True,
) -> float:
    """1.0 if answer matches expected after normalization, 0.0 otherwise.

    By default normalizes both sides (lowercase, strip punctuation and
    articles). Set ``normalize_text=False`` for a raw comparison that
    still strips whitespace and lowercases.

    Args:
        answer: The agent's answer (coerced to str if needed).
        expected: The expected string.
        normalize_text: Apply full normalization (default True).
    """
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
    """1.0 if the first number in the answer matches expected (within tolerance).

    Extracts the first numeric value from the answer string. Handles
    integers, decimals, and negative numbers.
    """
    a = str(answer) if not isinstance(answer, str) else answer
    match = re.search(r"-?\d+\.?\d*", a)
    if not match:
        return 0.0

    try:
        found = float(match.group())
    except ValueError:
        return 0.0

    return 1.0 if abs(found - expected) <= tolerance else 0.0


# =============================================================================
# Token-level metrics
# =============================================================================


def _tokenize(text: str) -> list[str]:
    """Tokenize normalized text into words."""
    return normalize(text).split()


def f1_score(
    answer: str | Any,
    reference: str,
) -> float:
    """Token-level F1 between answer and reference.

    Normalizes both texts, tokenizes into words, then computes
    precision, recall, and their harmonic mean. Returns 0.0 if
    either text is empty after normalization.

    Useful for free-form answers that are "mostly right" but may
    include extra or missing words compared to the reference.

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


# =============================================================================
# Multi-criterion grading
# =============================================================================


async def checklist(
    *checks: tuple[str, bool | Awaitable[bool]],
    weights: list[float] | None = None,
) -> ScenarioResult:
    """Score based on multiple checks, run in parallel when async.

    Each check is a ``(name, value)`` tuple where ``value`` is either a
    bool or an awaitable that returns a bool.  Awaitables (coroutines,
    tasks, futures) are gathered concurrently via ``asyncio.gather``,
    so long-running checks like DB queries or API calls execute in
    parallel instead of sequentially.

    Returns a ScenarioResult with subscores. Without explicit weights,
    all checks are weighted equally.

    Args:
        *checks: Tuples of ``(name, passed_or_coroutine)``.
        weights: Optional list of weights (should sum to ~1.0).
            Must match the number of checks if provided.

    Example::

        yield await checklist(
            ("cart_updated", product_in_cart(product)),  # coroutine
            ("order_placed", order_completed(product)),  # coroutine
            ("has_answer", bool(answer)),  # plain bool
            weights=[0.3, 0.5, 0.2],
        )
    """
    import asyncio

    n = len(checks)
    if n == 0:
        return ScenarioResult(reward=0.0, done=True, content="No checks provided")

    if weights is not None:
        if len(weights) != n:
            raise ValueError(
                f"Number of weights ({len(weights)}) must match number of checks ({n})"
            )
        w = weights
    else:
        w = [1.0 / n] * n

    names: list[str] = []
    awaitables: list[tuple[int, Awaitable[bool]]] = []
    resolved: dict[int, bool] = {}

    for i, (name, value) in enumerate(checks):
        names.append(name)
        if isinstance(value, Awaitable):
            awaitables.append((i, value))
        else:
            resolved[i] = bool(value)

    if awaitables:
        results = await asyncio.gather(*(aw for _, aw in awaitables))
        for (i, _), result in zip(awaitables, results, strict=True):
            resolved[i] = bool(result)

    subscores = [
        SubScore(name=names[i], weight=w[i], value=1.0 if resolved[i] else 0.0) for i in range(n)
    ]

    reward = sum(s.value * s.weight for s in subscores)

    passed_names = [names[i] for i in range(n) if resolved[i]]
    failed_names = [names[i] for i in range(n) if not resolved[i]]
    parts: list[str] = []
    if passed_names:
        parts.append("Passed: " + ", ".join(passed_names))
    if failed_names:
        parts.append("Failed: " + ", ".join(failed_names))
    content = "; ".join(parts)

    return ScenarioResult(
        reward=reward,
        done=True,
        content=content,
        subscores=subscores,
    )


# =============================================================================
# LLM-as-judge
# =============================================================================


async def llm_judge(
    answer: str | Any,
    *criteria: str | tuple[str, float],
    question: str = "",
    model: str = "claude-haiku-4-5",
) -> ScenarioResult:
    """Grade an answer against rubric criteria using an LLM judge.

    Each criterion is evaluated independently. The LLM decides pass/fail
    for each, and the final score is the weighted average. Requires the
    ``rubric`` package (``pip install rubric``).

    Uses the HUD inference gateway by default (``inference.hud.ai``).
    Requires ``HUD_API_KEY`` to be set.

    Args:
        answer: The agent's answer to grade.
        *criteria: Rubric criteria. Each is either a plain string
            (weight=1) or a ``(requirement, weight)`` tuple.
        question: The original question/prompt (provides context to the judge).
        model: Model to use for judging.

    Example::

        yield await llm_judge(
            answer,
            "Mentions at least 3 specific data points",
            ("Cites primary sources", 2.0),
            "Conclusion follows from evidence",
            question=prompt,
        )
    """
    try:
        from rubric import Criterion, Rubric
        from rubric.autograders import PerCriterionGrader
    except ImportError:
        raise ImportError(
            "llm_judge requires the 'rubric' package. Install with: pip install rubric"
        ) from None

    import os

    from openai import AsyncOpenAI

    parsed_criteria: list[Criterion] = []
    for c in criteria:
        if isinstance(c, tuple):
            req, weight = c
            parsed_criteria.append(Criterion(requirement=req, weight=weight))
        else:
            parsed_criteria.append(Criterion(requirement=c, weight=1.0))

    if not parsed_criteria:
        return ScenarioResult(reward=0.0, done=True, content="No criteria provided")

    api_key = os.environ.get("HUD_API_KEY", "")
    client = AsyncOpenAI(
        base_url="https://inference.hud.ai",
        api_key=api_key,
    )

    async def _generate(system_prompt: str, user_prompt: str) -> str:
        response = await client.chat.completions.create(
            model=model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""

    rubric_obj = Rubric(parsed_criteria)
    grader = PerCriterionGrader(generate_fn=_generate)
    result = await rubric_obj.grade(
        query=question,
        to_grade=str(answer),
        autograder=grader,
    )

    subscores = [
        SubScore(
            name=item.requirement[:50],
            weight=item.weight / sum(abs(c.weight) for c in parsed_criteria),
            value=1.0 if item.verdict else 0.0,
        )
        for item in result.report
    ]

    report_lines = [
        f"{'PASS' if item.verdict else 'FAIL'} {item.requirement}" for item in result.report
    ]
    content = "\n".join(report_lines)

    return ScenarioResult(
        reward=result.score,
        done=True,
        content=content,
        subscores=subscores if subscores else None,
    )


__all__ = [
    "checklist",
    "contains",
    "contains_all",
    "contains_any",
    "exact_match",
    "f1_score",
    "llm_judge",
    "normalize",
    "numeric_match",
]
