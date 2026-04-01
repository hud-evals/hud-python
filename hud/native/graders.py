<<<<<<< HEAD
"""Native graders for HUD evaluation.

All graders are async. ``Grade.gather`` runs them in parallel and
combines the results into an ``EvaluationResult`` you can yield
directly from a scenario.

Usage::

    from hud.native.graders import BashGrader, Grade, LLMJudgeGrader
    from hud.native.graders import exact_match, contains
    from hud.tools.types import SubScore

    # Simple one-liner
    yield exact_match(answer, "France")

    # Composed — all graders run in parallel
    yield await Grade.gather(
        BashGrader.grade(weight=0.5, command="pytest -q"),
        LLMJudgeGrader.grade(weight=0.3, answer=answer, criteria=["Correct"]),
        SubScore(name="format", value=exact_match(answer, "42"), weight=0.2),
    )
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable
=======
"""Generic graders for native HUD evaluation."""

from __future__ import annotations

import logging
import subprocess
from typing import Any
>>>>>>> 3bae301c628a4de571c2531d1815203ff104de20

from hud.tools.types import EvaluationResult, SubScore
from hud.utils.serialization import json_safe_dict

logger = logging.getLogger(__name__)

<<<<<<< HEAD

# =============================================================================
# Grade — the combiner
# =============================================================================
=======
__all__ = ["BashGrader", "Grade", "Grader"]
>>>>>>> 3bae301c628a4de571c2531d1815203ff104de20


def _dedupe_subscore_names(subscores: list[SubScore]) -> list[str]:
    """Return stable, unique names for a sequence of subscores."""
    name_counts: dict[str, int] = {}
    for item in subscores:
        name_counts[item.name] = name_counts.get(item.name, 0) + 1

    reserved_names = {item.name for item in subscores}
    name_usage: dict[str, int] = {}
    used_names: set[str] = set()
    final_names: list[str] = []

    for item in subscores:
        if name_counts[item.name] == 1 and item.name not in used_names:
            final_name = item.name
        else:
            suffix = name_usage.get(item.name, 0)
            while True:
                suffix += 1
                candidate = f"{item.name}-{suffix}"
                if candidate in used_names:
                    continue
                if candidate in reserved_names:
                    continue
                name_usage[item.name] = suffix
                final_name = candidate
                break
        used_names.add(final_name)
        final_names.append(final_name)

    return final_names


class Grade:
<<<<<<< HEAD
    """Combine ``SubScore`` items into a yieldable ``EvaluationResult``."""

    @staticmethod
    def from_subscores(subscores: list[SubScore]) -> EvaluationResult:
        """Combine already-resolved subscores into a weighted result.

        Positive weights are normalized to sum to ``1.0``.
        Negative weights are preserved as penalties.
        """
=======
    """Factory for building ``EvaluationResult`` objects from ``SubScore`` items."""

    @staticmethod
    def from_subscores(subscores: list[SubScore]) -> EvaluationResult:
        """Combine subscores into a weighted reward and ready-to-yield result.

        Positive weights are normalized to sum to ``1.0`` so the returned
        ``EvaluationResult`` lines up with the SDK's subscore semantics.
        Negative weights are preserved as penalties, including when they drive
        the final reward below zero.
        """

>>>>>>> 3bae301c628a4de571c2531d1815203ff104de20
        if not subscores:
            raise ValueError("subscores must not be empty")

        positive_weight_sum = sum(item.weight for item in subscores if item.weight > 0)
        if positive_weight_sum <= 0:
            raise ValueError("subscores must include at least one positive weight")

        normalized_subscores: list[SubScore] = []
        metadata: dict[str, Any] = {}

        for item, final_name in zip(subscores, _dedupe_subscore_names(subscores), strict=True):
            normalized_weight = (
                item.weight / positive_weight_sum if item.weight > 0 else item.weight
            )
            normalized_subscores.append(
                SubScore(
                    name=final_name,
                    weight=normalized_weight,
                    value=item.value,
                    metadata=item.metadata,
                )
            )
            if item.metadata is not None:
                metadata[final_name] = item.metadata

        reward = float(sum(item.value * item.weight for item in normalized_subscores))

        return EvaluationResult(
            reward=reward,
            done=True,
            subscores=normalized_subscores,
            info=metadata,
        )

<<<<<<< HEAD
    @staticmethod
    async def gather(*items: SubScore | Awaitable[SubScore]) -> EvaluationResult:
        """Resolve subscores and grader coroutines in parallel, then combine.

        Accepts a mix of:
        - ``SubScore`` objects (used immediately)
        - Awaitables returning ``SubScore`` (e.g. ``Grader.grade()``)

        All awaitables run concurrently via ``asyncio.gather``.

        Example::

            yield await Grade.gather(
                BashGrader.grade(weight=0.3, command="pytest -q"),
                LLMJudgeGrader.grade(weight=0.4, answer=answer, criteria=[...]),
                SubScore(name="answer", value=exact_match(answer, "42"), weight=0.3),
            )
        """
        from collections.abc import Awaitable as _Awaitable

        resolved: list[SubScore] = []
        pending: list[tuple[int, _Awaitable[SubScore]]] = []

        for item in items:
            if isinstance(item, SubScore):
                resolved.append(item)
            elif isinstance(item, _Awaitable):
                pending.append((len(resolved), item))
                resolved.append(SubScore(name="__placeholder__", value=0.0, weight=0.0))
            else:
                raise TypeError(
                    f"Expected SubScore or Awaitable[SubScore], "
                    f"got {type(item).__name__}"
                )

        if pending:
            results = await asyncio.gather(*(aw for _, aw in pending))
            for (slot, _), result in zip(pending, results, strict=True):
                resolved[slot] = result

        return Grade.from_subscores(resolved)


# =============================================================================
# Grader — async base class
# =============================================================================


class Grader:
    """Async base class for reusable graders.

    Subclasses implement ``compute_score`` (async). The ``grade`` classmethod
    calls it, wraps the result as a ``SubScore``, and records parameters
    in metadata for reproducibility.
    """
=======

class Grader:
    """Base class for reusable graders that emit ``SubScore`` objects."""
>>>>>>> 3bae301c628a4de571c2531d1815203ff104de20

    name: str = "BaseGrader"

    @classmethod
<<<<<<< HEAD
    async def grade(cls, weight: float, name: str | None = None, **kwargs: Any) -> SubScore:
        """Run the grader and package the result as a ``SubScore``."""
        result = await cls.compute_score(**kwargs)
=======
    def grade(cls, weight: float, name: str | None = None, **kwargs: Any) -> SubScore:
        """Run the grader and package the result as a ``SubScore``."""
        result = cls.compute_score(**kwargs)
>>>>>>> 3bae301c628a4de571c2531d1815203ff104de20

        if isinstance(result, tuple):
            score, metadata = result
        else:
            score = result
            metadata = {}

        return SubScore(
            name=name or cls.name,
            weight=weight,
            value=float(score),
            metadata={**metadata, "_parameters": json_safe_dict(kwargs)},
        )

    @classmethod
<<<<<<< HEAD
    async def compute_score(cls, **kwargs: Any) -> float | tuple[float, dict[str, Any]]:
        """Compute a score between ``0.0`` and ``1.0``.

        Return a float, or ``(float, metadata_dict)`` to attach extra info.
        """
=======
    def compute_score(cls, *args: Any, **kwargs: Any) -> float | tuple[float, dict[str, Any]]:
        """Compute a score between ``0.0`` and ``1.0``."""
>>>>>>> 3bae301c628a4de571c2531d1815203ff104de20
        raise NotImplementedError("Subclasses must implement compute_score")

    @classmethod
    def any(cls, weight: float, subscores: list[SubScore]) -> SubScore:
<<<<<<< HEAD
        """Subscore that passes if any input passes (max)."""
=======
        """Return a subscore that passes if any input subscore passes."""
>>>>>>> 3bae301c628a4de571c2531d1815203ff104de20
        if not subscores:
            raise ValueError("subscores must not be empty")

        unique_names = _dedupe_subscore_names(subscores)
        return SubScore(
            name=f"{cls.name}_any",
            value=max(subscore.value for subscore in subscores),
            weight=weight,
            metadata={
                "subscores": unique_names,
                "subscore_metadata": {
                    unique_name: subscore.metadata
                    for unique_name, subscore in zip(unique_names, subscores, strict=True)
                    if subscore.metadata is not None
                },
            },
        )

    @classmethod
    def all(cls, weight: float, subscores: list[SubScore]) -> SubScore:
<<<<<<< HEAD
        """Subscore that passes only if all inputs pass (min)."""
=======
        """Return a subscore that passes only if all input subscores pass."""
>>>>>>> 3bae301c628a4de571c2531d1815203ff104de20
        if not subscores:
            raise ValueError("subscores must not be empty")

        unique_names = _dedupe_subscore_names(subscores)
        return SubScore(
            name=f"{cls.name}_all",
            value=min(subscore.value for subscore in subscores),
            weight=weight,
            metadata={
                "subscores": unique_names,
                "subscore_metadata": {
                    unique_name: subscore.metadata
                    for unique_name, subscore in zip(unique_names, subscores, strict=True)
                    if subscore.metadata is not None
                },
            },
        )


<<<<<<< HEAD
# =============================================================================
# BashGrader — async subprocess
# =============================================================================


class BashGrader(Grader):
    """Run a shell command and score by exit code. Fully async."""
=======
class BashGrader(Grader):
    """Run a shell command and score it by exit code."""
>>>>>>> 3bae301c628a4de571c2531d1815203ff104de20

    name = "BashGrader"

    @classmethod
<<<<<<< HEAD
    async def compute_score(
        cls,
        command: str,
        cwd: str | None = None,
        timeout_seconds: int = 60,
        **kwargs: Any,
    ) -> tuple[float, dict[str, Any]]:
        """Run ``command`` via ``bash -lc`` and return score + metadata."""
        del kwargs
        logger.info(
            "Running grader command: %s (cwd=%s, timeout=%ss)", command, cwd, timeout_seconds
        )
        try:
            proc = await asyncio.create_subprocess_exec(
                "/bin/bash", "-lc", command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_seconds
            )
            stdout = stdout_bytes.decode(errors="replace")
            stderr = stderr_bytes.decode(errors="replace")
            returncode = proc.returncode or 0
        except TimeoutError:
            proc.kill()
=======
    def compute_score(
        cls,
        command: str,
        cwd: str | None = None,
        timeout: int = 60,
        **kwargs: Any,
    ) -> tuple[float, dict[str, Any]]:
        """Run ``command`` via ``bash -lc`` and return score plus execution metadata."""
        del kwargs
        logger.info("Running grader command: %s (cwd=%s, timeout=%ss)", command, cwd, timeout)
        try:
            result = subprocess.run(
                ["/bin/bash", "-lc", command],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = (
                (exc.stdout or b"").decode(errors="replace")
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or "")
            )
            stderr = (
                (exc.stderr or b"").decode(errors="replace")
                if isinstance(exc.stderr, bytes)
                else (exc.stderr or "")
            )
>>>>>>> 3bae301c628a4de571c2531d1815203ff104de20
            return (
                0.0,
                {
                    "exit_code": None,
<<<<<<< HEAD
                    "stdout": "",
                    "stderr": "",
                    "timed_out": True,
                    "timeout": timeout_seconds,
                },
            )
        except FileNotFoundError:
            return (
                0.0,
                {
                    "exit_code": None,
                    "stdout": "",
                    "stderr": "/bin/bash not found",
                    "timed_out": False,
                },
            )

        score = 1.0 if returncode == 0 else 0.0
        return (score, {"exit_code": returncode, "stdout": stdout, "stderr": stderr})


# =============================================================================
# LLMJudgeGrader — rubric-based LLM evaluation
# =============================================================================


class LLMJudgeGrader(Grader):
    """Grade an answer against rubric criteria using an LLM judge.

    Requires the ``rubric`` package (``pip install rubric``).
    Uses the HUD inference gateway by default.

    Example::

        yield await Grade.gather(
            BashGrader.grade(weight=0.4, command="pytest -q"),
            LLMJudgeGrader.grade(
                weight=0.6,
                answer=answer,
                criteria=["Correct", ("Well-reasoned", 2.0)],
                question=prompt,
            ),
        )
    """

    name = "LLMJudgeGrader"

    @classmethod
    async def compute_score(
        cls,
        answer: str | Any = "",
        criteria: list[str | tuple[str, float]] | None = None,
        question: str = "",
        model: str = "claude-haiku-4-5",
        **kwargs: Any,
    ) -> tuple[float, dict[str, Any]]:
        """Evaluate answer against criteria via LLM."""
        del kwargs
        try:
            from rubric import Criterion, Rubric
            from rubric.autograders import PerCriterionGrader
        except ImportError:
            raise ImportError(
                "LLMJudgeGrader requires the 'rubric' package. "
                "Install with: pip install rubric"
            ) from None

        import os

        from openai import AsyncOpenAI

        parsed: list[Criterion] = []
        for c in criteria or []:
            if isinstance(c, tuple):
                req, w = c
                parsed.append(Criterion(requirement=req, weight=w))
            else:
                parsed.append(Criterion(requirement=c, weight=1.0))

        if not parsed:
            return (0.0, {"error": "no criteria provided"})

        api_key = os.environ.get("HUD_API_KEY", "")
        client = AsyncOpenAI(base_url="https://inference.hud.ai", api_key=api_key)

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

        rubric_obj = Rubric(parsed)
        autograder = PerCriterionGrader(generate_fn=_generate)
        result = await rubric_obj.grade(
            query=question,
            to_grade=str(answer),
            autograder=autograder,
        )

        verdicts = {
            item.requirement[:80]: {
                "verdict": item.verdict,
                "reason": getattr(item, "reason", None),
                "weight": item.weight,
            }
            for item in result.report
        }

        return (float(result.score), {"criteria": verdicts, "model": model})


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
# Answer comparisons (return float for use as SubScore.value)
# =============================================================================


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
    "BashGrader",
    "Grade",
    "Grader",
    "LLMJudgeGrader",
    "contains",
    "contains_all",
    "contains_any",
    "exact_match",
    "f1_score",
    "normalize",
    "numeric_match",
]
=======
                    "stdout": stdout,
                    "stderr": stderr,
                    "timed_out": True,
                    "timeout": timeout,
                },
            )

        score = 1.0 if result.returncode == 0 else 0.0
        return (
            score,
            {
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )
>>>>>>> 3bae301c628a4de571c2531d1815203ff104de20
