"""``LLMJudgeGrader`` — rubric-based LLM evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from .base import Grader

if TYPE_CHECKING:
    from openai import AsyncOpenAI


class LLMJudgeGrader(Grader):
    """Grade an answer against rubric criteria using an LLM judge.

    Requires the ``rubric`` package (``pip install rubric``).
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
        **kwargs: Any,
    ) -> tuple[float, dict[str, Any]]:
        """Evaluate answer against criteria via LLM."""
        del kwargs
        try:
            from rubric import Criterion, Rubric
            from rubric.autograders import PerCriterionGrader
        except ImportError:
            raise ImportError(
                "LLMJudgeGrader requires the 'rubric' package. Install with: pip install rubric"
            ) from None

        from hud.utils.gateway import build_gateway_client

        parsed: list[Criterion] = []
        for c in criteria or []:
            if isinstance(c, tuple):
                req, w = c
                parsed.append(Criterion(requirement=req, weight=w))
            else:
                parsed.append(Criterion(requirement=c, weight=1.0))

        if not parsed:
            return (0.0, {"error": "no criteria provided"})

        client = cast("AsyncOpenAI", build_gateway_client("openai"))

        async def _generate(system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
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
            for item in (result.report or [])
        }

        return (float(result.score), {"criteria": verdicts, "model": model})


__all__ = ["LLMJudgeGrader"]
