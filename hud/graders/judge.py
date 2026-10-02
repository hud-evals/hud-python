"""``LLMJudgeGrader``: per-criterion LLM evaluation through the HUD inference gateway."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from .base import Grader
from .results import SubScore

if TYPE_CHECKING:
    from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# A whole uppercase MET; an UN or NOT prefix ("not" in any case) makes it UNMET.
_VERDICT = re.compile(r"\b(UN[_-]?|(?i:not)[\s_-]+)?MET\b")

_SYSTEM_PROMPT = """You evaluate a response against a single criterion and decide \
whether the thing the criterion describes is present in the response.

The <criterion_type> field says whether the criterion describes something desirable \
(positive) or an error to avoid (negative). Your job is the same for both: decide if \
the thing described is actually present.

- positive criterion -> MET when the response contains/satisfies it, UNMET otherwise.
- negative criterion -> MET when the response actually makes the error, UNMET when it \
does not (or only mentions it to warn against it).

Rules:
- Be strict about factual accuracy but flexible about wording; accept semantically \
equivalent statements and reasonable implications.
- Watch for negation, warnings, and contrasts ("unlike X...", "avoid X").
- An action required "immediately"/unconditionally is UNMET if the response only does \
it conditionally ("if Y, then ...").
- "criterion_status" is about presence, not quality.

Return ONLY raw JSON, no code fences, in exactly this form:
{"criterion_status": "MET", "explanation": "Brief reason."}"""


@dataclass(slots=True)
class _Criterion:
    requirement: str
    weight: float


class LLMJudgeGrader(Grader):
    """Grade an answer against weighted criteria with an LLM judge on the HUD gateway.

    Criteria are judged concurrently, one call each. A negative weight marks an error the
    answer must not make, and counts as a penalty when the judge finds it.
    """

    name = "LLMJudgeGrader"

    @classmethod
    async def compute_score(
        cls,
        answer: str | Any = "",
        criteria: list[str | tuple[str, float]] | None = None,
        question: str = "",
        model: str = "claude-sonnet-5-5",
        **kwargs: Any,
    ) -> SubScore:
        """Evaluate ``answer`` against ``criteria`` via parallel LLM judgments."""
        del kwargs
        parsed = _parse_criteria(criteria)
        if not parsed:
            return SubScore(name=cls.name, value=0.0, info={"error": "no criteria provided"})

        from hud.utils.gateway import build_gateway_client

        client = cast("AsyncOpenAI", build_gateway_client("openai"))
        answer_text = str(answer)

        async def _judge(criterion: _Criterion) -> SubScore:
            response = await client.chat.completions.create(
                model=model,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": _user_prompt(criterion, answer_text, question)},
                ],
            )
            met, reason = _parse_verdict(response.choices[0].message.content or "")
            return SubScore(
                name=criterion.requirement,
                value=1.0 if met else 0.0,
                weight=criterion.weight,
                info={"reason": reason},
            )

        verdicts = list(await asyncio.gather(*(_judge(c) for c in parsed)))
        return SubScore(
            name=cls.name,
            value=_rubric_value(verdicts),
            children=verdicts,
            info={"model": model},
        )


def _parse_criteria(criteria: list[str | tuple[str, float]] | None) -> list[_Criterion]:
    parsed: list[_Criterion] = []
    for item in criteria or []:
        if isinstance(item, tuple):
            requirement, weight = item
            parsed.append(_Criterion(str(requirement), float(weight)))
        else:
            parsed.append(_Criterion(str(item), 1.0))
    return parsed


def _user_prompt(criterion: _Criterion, answer: str, question: str) -> str:
    criterion_type = "negative" if criterion.weight < 0 else "positive"
    query = f"<query>\n{question}\n</query>\n\n" if question else ""
    return (
        f"<criterion_type>\n{criterion_type}\n</criterion_type>\n\n"
        f"<criterion>\n{criterion.requirement}\n</criterion>\n\n"
        f"{query}"
        f"<response>\n{answer}\n</response>"
    )


def _parse_verdict(content: str) -> tuple[bool, str]:
    """Extract ``(met, explanation)`` from the judge's reply; raise if it has no verdict."""
    text = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
        except ValueError:
            logger.debug("LLMJudgeGrader: unparseable judge reply: %s", text[:200])
        else:
            status = str(data.get("criterion_status")).upper()
            if status not in ("MET", "UNMET"):
                raise ValueError(
                    f"LLMJudgeGrader: judge reply has no MET/UNMET criterion_status: {text[:200]!r}"
                )
            return status == "MET", str(data.get("explanation", ""))
    negations = _VERDICT.findall(text)
    if not negations:
        raise ValueError(f"LLMJudgeGrader: judge reply has no MET/UNMET verdict: {text[:200]!r}")
    return not any(negations), text[:200]


def _rubric_value(subscores: list[SubScore]) -> float:
    """Aggregate criterion verdicts into a rubric score."""
    total_positive = sum(max(0.0, subscore.weight) for subscore in subscores)
    total_negative = sum(abs(subscore.weight) for subscore in subscores if subscore.weight < 0)
    weighted_sum = sum(subscore.value * subscore.weight for subscore in subscores)
    if total_positive > 0:
        return max(0.0, min(1.0, weighted_sum / total_positive))
    if total_negative > 0:
        return max(0.0, min(1.0, 1.0 + weighted_sum / total_negative))
    return 0.0
