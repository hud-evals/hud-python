"""A virtual Linux desktop exposed through an RFB capability."""

# NOTE: do NOT add `from __future__ import annotations` here - under it a typed @env.template
# param crashes the sync/deploy manifest path (TypeAdapter on a string forward-ref). Keep
# annotations as real objects.
import asyncio
from collections.abc import Awaitable
from typing import Any

from hud import Environment
from hud.capabilities import Capability
from hud.graders import BashGrader, LLMJudgeGrader, SubScore, combine
from hud.settings import settings

env = Environment(name="cua")  # literal name - `hud deploy` static-parses it
_task_started = False

_HOST = "127.0.0.1"
_VNC_PORT = 5900


async def _listening(host: str, port: int, timeout: float = 30.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            _, writer = await asyncio.open_connection(host, port)
        except OSError:
            await asyncio.sleep(0.2)
        else:
            writer.close()
            await writer.wait_closed()
            return
    raise RuntimeError(f"VNC server never came up on {host}:{port}")


@env.initialize
async def _up() -> None:
    await _listening(_HOST, _VNC_PORT)
    env.add_capability(Capability.rfb(name="screen", url=f"rfb://{_HOST}", display=0))


def make_prompt(description: str) -> str:
    """Format a task description into an agent prompt."""
    return f"Use computer use tools to complete the following task:\n\n{description}"


@env.template()
async def cua_task(
    prompt: str,
    bash_checks: list[dict[str, Any]] | None = None,
    grading_criteria: list[str] | None = None,
):
    """General CUA task: present the prompt, then grade with any combination of deterministic
    bash checks (run server-side in this container) and an LLM rubric judge.

    `combine` normalizes the positive weights to sum to 1.0, so weights are relative.

    Args:
        prompt: The task instruction shown to the agent.
        bash_checks: Optional list of {"name", "command", "weight"} for shell-based grading.
        grading_criteria: Optional rubric strings for the LLM judge (needs HUD_API_KEY).
    """
    global _task_started

    if any(c.get("weight", 1.0) < 0 for c in (bash_checks or [])):
        raise ValueError("bash check weights must be nonnegative")
    bash_total = sum(c.get("weight", 1.0) for c in (bash_checks or []))
    if bash_checks and bash_total <= 0:
        raise ValueError("bash check weights must sum to a positive value")
    if grading_criteria and not settings.api_key:
        raise RuntimeError("HUD_API_KEY is required for CUA tasks with grading_criteria")
    if _task_started:
        raise RuntimeError("CUA supports one task per substrate; start a fresh runtime")
    _task_started = True

    answer = yield make_prompt(prompt)
    bash_share = 0.5 if grading_criteria else 1.0

    graders: list[SubScore | Awaitable[SubScore]] = []

    for check in bash_checks or []:
        graders.append(
            BashGrader.grade(
                weight=check.get("weight", 1.0) / bash_total * bash_share,
                name=check["name"],
                command=check["command"],
            )
        )

    if grading_criteria:
        judge_weight = 0.5 if bash_checks else 1.0
        graders.append(
            LLMJudgeGrader.grade(
                weight=judge_weight,
                name="llm_judge",
                answer=str(answer),
                question=prompt,
                criteria=[(c, 1.0) for c in grading_criteria],
            )
        )

    if not graders:
        graders.append(SubScore(name="desktop_running", value=1.0, weight=1.0))

    yield await combine(*graders)
