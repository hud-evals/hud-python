"""A virtual Linux desktop exposed through an RFB capability."""

# NOTE: do NOT add `from __future__ import annotations` here - under it a typed @env.template
# param crashes the sync/deploy manifest path (TypeAdapter on a string forward-ref). Keep
# annotations as real objects.
import asyncio
import logging

from hud import Environment
from hud.capabilities import Capability
from hud.graders import BashGrader, LLMJudgeGrader, SubScore, combine
from hud.settings import settings

logger = logging.getLogger(__name__)

env = Environment(name="cua")  # literal name - `hud deploy` static-parses it

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
    bash_checks: list[dict] | None = None,
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
    answer = yield make_prompt(prompt)

    bash_total = sum(c.get("weight", 1.0) for c in (bash_checks or []))
    if bash_checks and bash_total <= 0:
        raise ValueError("bash check weights must sum to a positive value")
    bash_share = 0.5 if grading_criteria else 1.0

    graders: list = []

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
        if settings.api_key:
            graders.append(
                LLMJudgeGrader.grade(
                    weight=judge_weight,
                    name="llm_judge",
                    answer=str(answer),
                    question=prompt,
                    criteria=[(c, 1.0) for c in grading_criteria],
                )
            )
        else:
            # No key (e.g. a keyless deploy): the judge can't run, so it scores 0 at its real
            # weight instead of erroring the trace. Never re-weight the bash checks.
            logger.warning("No HUD_API_KEY: LLM judge skipped; it scores 0 at its weight.")
            graders.append(SubScore(name="llm_judge", weight=judge_weight, value=0.0))

    if not graders:
        graders.append(SubScore(name="desktop_running", value=1.0, weight=1.0))

    yield await combine(*graders)
