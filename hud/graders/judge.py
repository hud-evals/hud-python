"""``LLMJudgeGrader`` — per-criterion LLM evaluation.

A self-contained implementation of weighted per-criterion judging: each criterion
is graded ``MET``/``UNMET`` by an LLM in parallel, and the verdicts are combined
by weight into a 0-1 score. No third-party dependency — it talks to the HUD
inference gateway directly.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from .base import Grader
from .results import SubScore

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletionMessageParam

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
- Include a non-empty explanation for the verdict.

Return ONLY raw JSON, no code fences, in exactly this form:
{"criterion_status": "MET", "explanation": "Brief reason."}"""

_REPAIR_PROMPT = """Repair only the format of your previous verdict. Keep the same judgment \
about the original criterion, query and response. Return ONLY one raw JSON object, no prose \
or code fences. Set criterion_status to MET or UNMET and explanation to a non-empty string: \
{"criterion_status": "MET", "explanation": "Brief reason."}"""


@dataclass(slots=True)
class _Criterion:
    requirement: str
    weight: float


class LLMJudgeGrader(Grader):
    """Grade an answer against weighted criteria using an LLM judge.

    Uses the HUD inference gateway by default.

    Example::

        yield await combine(
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
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> SubScore:
        """Evaluate ``answer`` against ``criteria`` via parallel LLM judgments."""
        del kwargs
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        parsed = _parse_criteria(criteria)
        if not parsed:
            return SubScore(name=cls.name, value=0.0, info={"error": "no criteria provided"})

        from hud.utils.gateway import build_gateway_client

        client = cast("AsyncOpenAI", build_gateway_client("openai"))
        answer_text = str(answer)

        async def _judge(criterion: _Criterion) -> SubScore:
            messages: list[ChatCompletionMessageParam] = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _user_prompt(criterion, answer_text, question)},
            ]
            attempts = 0
            while True:
                attempts += 1
                response = await client.chat.completions.create(
                    model=model, max_tokens=max_tokens, messages=messages
                )
                choice = response.choices[0]
                if choice.finish_reason == "length":
                    raise RuntimeError(
                        f"LLM judge {model} exhausted max_tokens={max_tokens} without "
                        "a complete verdict"
                    )
                content = choice.message.content or ""
                try:
                    met, reason = _parse_verdict(content)
                except ValueError:
                    if attempts == 2:
                        raise
                    messages = [
                        *messages,
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": _REPAIR_PROMPT},
                    ]
                    continue
                return SubScore(
                    name=criterion.requirement,
                    value=1.0 if met else 0.0,
                    weight=criterion.weight,
                    info={"reason": reason, "attempts": attempts},
                )

        verdicts = list(await asyncio.gather(*(_judge(c) for c in parsed)))
        return SubScore(
            name=cls.name,
            value=_rubric_value(verdicts),
            children=verdicts,
            info={"model": model, "max_tokens": max_tokens},
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
    """Require a structured verdict; missing judgments must never become UNMET."""
    text = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        try:
            data = json.loads(text[start : end + 1] if start != -1 and end > start else "")
        except ValueError:
            raise ValueError("LLM judge returned an invalid structured verdict") from None
    if (
        not isinstance(data, dict)
        or data.get("criterion_status") not in ("MET", "UNMET")
        or not isinstance(data.get("explanation"), str)
        or not data["explanation"].strip()
    ):
        raise ValueError("LLM judge returned an invalid structured verdict")
    return data["criterion_status"] == "MET", data["explanation"]


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


__all__ = ["LLMJudgeGrader"]
