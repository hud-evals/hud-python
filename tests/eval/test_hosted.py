"""Hosted execution: the platform runs the whole rollout; this process submits it,
polls its trace to a terminal state, and folds the result into a ``Run``. A deadline
or a cancel here cancels the rollout on the platform."""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from typing import TYPE_CHECKING, Any

import pytest
from dirty_equals import IsStr

from hud.agents.claude import ClaudeCLIAgent, ClaudeCLIConfig
from hud.agents.openai_compatible import OpenAIChatAgent
from hud.agents.types import OpenAIChatConfig
from hud.eval import (
    ComposeProject,
    HostedRuntime,
    RuntimeConfig,
    Task,
    Taskset,
)
from tests.support import Reply, eventually

if TYPE_CHECKING:
    from pathlib import Path

    from hud.agents.base import Agent
    from tests.support import Platform
    from tests.support.platform import Trace

SUBMIT = "/v2/rollouts/submit"
TRACE_ID = "1" * 32
JOB_ID = "2" * 32
ROW = Task(env="sums", id="add", args={"a": 1, "b": 2})


def chat_agent() -> OpenAIChatAgent:
    return OpenAIChatAgent(
        OpenAIChatConfig(model="test-model", api_key="secret", base_url="http://model.test")
    )


def hosted(platform: Platform) -> Trace:
    """The one trace the platform was asked to run."""
    (trace,) = platform.traces.values()
    return trace


def submitted(platform: Platform) -> dict[str, Any]:
    body = hosted(platform).submitted
    assert body is not None
    return body


async def test_a_hosted_rollout_submits_the_row_and_agent_then_polls_to_completion(
    platform: Platform, tmp_path: Path
) -> None:
    compose = tmp_path / "compose.yaml"
    compose.write_text("services:\n  main:\n    image: sums:latest\n")
    platform.next_rollout = [
        {"status": "pending"},
        {"status": "running"},
        {"status": "completed", "reward": 0.5},
    ]
    row = Task(
        env="sums",
        id="add",
        agent_config={"timeout_seconds": 45.0},
        runtime_config=RuntimeConfig(compose=ComposeProject(document=compose)),
        verifier=Task(env="judge", id="verify", runtime_config=RuntimeConfig(image="judge:1")),
    )

    run = await HostedRuntime(poll_interval=0).run(
        row, chat_agent(), job_id=JOB_ID, trace_id=TRACE_ID
    )

    body = submitted(platform)
    assert (body["trace_id"], body["job_id"]) == (str(uuid.UUID(TRACE_ID)), str(uuid.UUID(JOB_ID)))
    # The platform gets the Compose document itself, and the row's agent timeout.
    document = body["runtime_config"]["compose"]["document"]
    assert document["services"]["main"]["image"] == "sums:latest"
    assert body["agent"]["config"]["timeout_seconds"] == 45.0
    # The verifier travels as its own row, so the platform gives it its own sandbox.
    assert (body["verifier"]["env"], body["verifier"]["runtime_config"]["image"]) == (
        "judge",
        "judge:1",
    )
    assert "secret" not in str(body)
    assert (run.reward, run.trace.status) == (0.5, "completed")


@pytest.mark.parametrize(
    ("agent", "registered"),
    [
        pytest.param(chat_agent(), "openai_compatible", id="a-chat-agent"),
        pytest.param(
            ClaudeCLIAgent(ClaudeCLIConfig(model="claude-sonnet-4-6", gateway=True)),
            "claude_cli",
            id="a-cli-agent",
        ),
    ],
)
async def test_a_hosted_rollout_sends_its_agent_as_its_registered_type(
    agent: Agent, registered: str, platform: Platform
) -> None:
    await HostedRuntime(poll_interval=0).run(ROW, agent, job_id=JOB_ID, trace_id=TRACE_ID)

    assert submitted(platform)["agent"]["type"] == registered


TERMINAL = {
    "a completed trace with its evaluation": (
        {
            "status": "completed",
            "reward": 0.5,
            "evaluation_result": {"score": 0.5, "content": "half", "info": {"k": 1}},
        },
        (0.5, "completed", None, {"score": 0.5, "content": "half", "info": {"k": 1}}, False),
    ),
    "an errored trace without a reward": (
        {"status": "error", "reward": None, "error": "env exploded"},
        (0.0, "error", "env exploded", {}, True),
    ),
    "an errored trace that was still graded": (
        {"status": "error", "reward": 0.75, "error": "agent exploded"},
        (0.75, "error", "agent exploded", {"score": 0.75}, False),
    ),
    "a trace cancelled before grading": (
        {"status": "cancelled", "reward": None},
        (0.0, "cancelled", None, {}, True),
    ),
    "a trace whose evaluation has no numeric score": (
        {"status": "completed", "evaluation_result": {"score": "high"}},
        (0.0, "error", IsStr(), {}, True),
    ),
}


@pytest.mark.parametrize(("state", "folded"), TERMINAL.values(), ids=TERMINAL.keys())
async def test_a_terminal_trace_folds_into_the_run_and_its_job(
    state: dict[str, Any], folded: tuple[Any, ...], platform: Platform
) -> None:
    platform.next_rollout = [state]

    job = await Taskset("hosted", [ROW]).run(chat_agent(), runtime=HostedRuntime(poll_interval=0))

    (run,) = job.runs
    assert (run.reward, run.trace.status, run.trace.error, run.grade.raw, run in job.errors) == (
        folded
    )


async def test_a_rejected_submission_is_a_failed_run(platform: Platform) -> None:
    platform.services.route(
        "api", "POST", SUBMIT, json={"detail": "unknown env 'sums'"}, status=422
    )

    job = await ROW.run(chat_agent(), runtime=HostedRuntime(poll_interval=0))

    (run,) = job.runs
    assert run.trace.status == "error"
    assert "unknown env 'sums'" in str(run.trace.error)
    assert platform.services.requests("api", "GET") == []


# ─── stopping a hosted rollout ─────────────────────────────────────────

DID_NOT_FINISH = IsStr(regex=r"hosted rollout [0-9a-f]{32} did not finish within 1s")


def cancelled_on_the_platform(platform: Platform) -> bool:
    return platform.traces != {} and all(trace.cancelled for trace in platform.traces.values())


@pytest.mark.parametrize(
    ("accepted", "deprecated"),
    [
        pytest.param(True, False, id="while-the-platform-runs-it"),
        pytest.param(False, False, id="before-the-platform-accepts-it"),
        pytest.param(True, True, id="set-by-the-deprecated-run_timeout"),
    ],
)
async def test_a_hosted_rollout_past_its_deadline_is_cancelled_on_the_platform(
    accepted: bool, deprecated: bool, platform: Platform
) -> None:
    platform.next_rollout = [{"status": "running"}]
    if not accepted:
        platform.services.route("api", "POST", SUBMIT, Reply(json={}, delay=5))
    warned = (
        pytest.warns(DeprecationWarning, match="rollout_timeout=... to Task.run")
        if deprecated
        else contextlib.nullcontext()
    )

    with warned:
        runtime = HostedRuntime(poll_interval=0, run_timeout=1.0 if deprecated else None)
        job = await ROW.run(
            chat_agent(), runtime=runtime, rollout_timeout=None if deprecated else 1.0
        )

    (run,) = job.runs
    assert (run.reward, run.trace.status, run.trace.stop_reason, run.trace.error) == (
        0.0,
        "error",
        "timeout",
        DID_NOT_FINISH,
    )
    await eventually(lambda: cancelled_on_the_platform(platform), within=10)


async def test_cancelling_a_hosted_rollout_cancels_it_on_the_platform(platform: Platform) -> None:
    platform.next_rollout = [{"status": "running"}]

    pending = asyncio.create_task(ROW.run(chat_agent(), runtime=HostedRuntime(poll_interval=0)))
    await eventually(lambda: platform.services.requests("api", "POST", SUBMIT) != [])
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    await eventually(lambda: cancelled_on_the_platform(platform), within=10)
