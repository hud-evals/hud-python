"""Native graders for HUD evaluation.

All graders are async. ``combine`` runs them in parallel and combines the
results into an ``EvaluationResult`` you can yield directly from a task::

    from hud.graders import BashGrader, LLMJudgeGrader, SubScore, combine

    # Composed — all graders run in parallel
    yield await combine(
        BashGrader.grade(weight=0.5, command="pytest -q"),
        LLMJudgeGrader.grade(weight=0.3, answer=answer, criteria=["Correct"]),
        SubScore(name="format", value=1.0 if answer.strip() == "42" else 0.0, weight=0.2),
    )

The package is split into focused modules (``results``, ``combine``, ``base``,
``bash``, ``judge``); import from ``hud.graders`` directly — the layout is an
implementation detail.
"""

from __future__ import annotations

from .base import Grader
from .bash import BashGrader
from .combine import _combine_subscores, combine, combine_all, combine_any
from .judge import LLMJudgeGrader
from .results import EvaluationResult, SubScore

__all__ = [
    "BashGrader",
    "EvaluationResult",
    "Grader",
    "LLMJudgeGrader",
    "SubScore",
    "_combine_subscores",
    "combine",
    "combine_all",
    "combine_any",
]
