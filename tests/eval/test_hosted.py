"""Hosted execution: the platform runs the whole rollout; this process submits it,
polls its trace to a terminal state, and folds the result into a ``Run``."""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any

import pytest
from dirty_equals import IsStr
from inline_snapshot import snapshot

from hud.agents.claude import ClaudeCLIAgent, ClaudeCLIConfig
from hud.agents.openai_compatible import OpenAIChatAgent
from hud.agents.types import OpenAIChatConfig
from hud.eval import (
    ComposeProject,
    HostedRuntime,
    RuntimeConfig,
    RuntimeGPU,
    RuntimeLimits,
    RuntimeResources,
    Task,
    Taskset,
)
from hud.telemetry.context import set_trace_context
from tests.eval.envs import eventually
from tests.harness import Reply, ScriptedAgent

if TYPE_CHECKING:
    from pathlib import Path

    from hud.agents.base import Agent
    from tests.harness import FakeServices, HudEnv

SUBMIT = "/v2/rollouts/submit"
CANCEL = "/v2/rollouts/cancel"
POLL = "/v2/trace/{id}"
TRACE_ID = "1" * 32
JOB_ID = "2" * 32
ROW = Task(env="sums", id="add", args={"a": 1, "b": 2})


@pytest.fixture
def platform(services: FakeServices, hud_env: HudEnv) -> FakeServices:
    """The fake platform accepting submissions, with a key set."""
    hud_env.set(HUD_API_KEY="k")
    services.route("api", "POST", SUBMIT, json={"status": "queued"})
    services.route("api", "POST", CANCEL, json={})
    return services


def chat_agent() -> OpenAIChatAgent:
    return OpenAIChatAgent(
        OpenAIChatConfig(model="test-model", api_key="secret", base_url="http://model.test")
    )


def submitted(platform: FakeServices) -> dict[str, Any]:
    (request,) = platform.requests("api", "POST", SUBMIT)
    assert request.bearer == "k"
    return request.json


async def test_a_hosted_rollout_submits_the_row_and_agent_then_polls_to_completion(
    platform: FakeServices, tmp_path: Path
) -> None:
    compose = tmp_path / "compose.yaml"
    compose.write_text("services:\n  main:\n    image: sums:latest\n")
    platform.route(
        "api",
        "GET",
        POLL,
        Reply(json={"status": "pending"}),
        Reply(json={"status": "running"}),
        Reply(json={"status": "completed", "reward": 0.5}),
    )
    row = Task(
        env="sums",
        id="add",
        slug="sums-add",
        args={"a": 1, "b": 2},
        agent_config={"timeout_seconds": 45.0},
        runtime_config=RuntimeConfig(
            compose=ComposeProject(document=compose, service_access=True),
            resources=RuntimeResources(cpu=2, gpu=RuntimeGPU(type="L4"), memory_mb=None),
            limits=RuntimeLimits(startup_timeout_s=120, run_timeout_s=900),
        ),
        verifier=Task(
            env="judge",
            id="verify",
            args={"expected": 3},
            runtime_config=RuntimeConfig(resources=None),
        ),
    )

    with set_trace_context("3" * 32):
        run = await HostedRuntime(poll_interval=0).run(
            row, chat_agent(), job_id=JOB_ID, group_id="g1", trace_id=TRACE_ID
        )

    assert submitted(platform) == snapshot(
        {
            "trace_id": "11111111-1111-1111-1111-111111111111",
            "job_id": "22222222-2222-2222-2222-222222222222",
            "env": "sums",
            "task": "add",
            "slug": "sums-add",
            "args": {"a": 1, "b": 2},
            "agent": {
                "type": "openai_compatible",
                "config": {
                    "timeout_seconds": 45.0,
                    "model_name": "OpenAI Chat",
                    "model": "test-model",
                    "gateway": False,
                    "auto_respond": False,
                    "max_steps": 10,
                    "tool_timeout_seconds": None,
                    "system_prompt": None,
                    "citations_enabled": False,
                    "stop_on": [],
                    "screenshot_encoding": {"mime_type": "image/png"},
                    "checkpoint": None,
                    "completion_kwargs": {},
                },
            },
            "group_id": "g1",
            "parent_trace_id": "33333333-3333-3333-3333-333333333333",
            "runtime_config": {
                "compose": {
                    "document": {
                        "services": {
                            "main": {
                                "image": "sums:latest",
                                "environment": {},
                                "expose": [],
                                "ports": [],
                                "volumes": [],
                            }
                        },
                        "networks": {},
                    },
                    "service_access": True,
                },
                "resources": {"cpu": 2.0, "gpu": {"type": "L4"}},
                "limits": {"startup_timeout_s": 120, "run_timeout_s": 900},
            },
            "verifier": {
                "env": "judge",
                "id": "verify",
                "args": {"expected": 3},
                "slug": "verify-5579a3e5",
                "runtime_config": {"resources": None},
            },
        }
    )
    polls = platform.requests("api", "GET", POLL)
    assert [poll.params["id"] for poll in polls] == [str(uuid.UUID(TRACE_ID))] * 3
    assert (run.reward, run.trace.status, run.trace_id, run.job_id, run.group_id, run.slug) == (
        0.5,
        "completed",
        TRACE_ID,
        JOB_ID,
        "g1",
        "sums-add",
    )
    assert run.runtime == f"hud://trace/{TRACE_ID}"
    with pytest.raises(RuntimeError, match="no live client"):
        run.client


AGENTS: dict[str, tuple[Agent, dict[str, Any]]] = {
    "a chat agent travels without its credentials": (
        chat_agent(),
        snapshot(
            {
                "type": "openai_compatible",
                "config": {
                    "timeout_seconds": None,
                    "model_name": "OpenAI Chat",
                    "model": "test-model",
                    "gateway": False,
                    "auto_respond": False,
                    "max_steps": 10,
                    "tool_timeout_seconds": None,
                    "system_prompt": None,
                    "citations_enabled": False,
                    "stop_on": [],
                    "screenshot_encoding": {"mime_type": "image/png"},
                    "checkpoint": None,
                    "completion_kwargs": {},
                },
            }
        ),
    ),
    "a CLI agent travels as its registered type": (
        ClaudeCLIAgent(ClaudeCLIConfig(model="claude-sonnet-4-6", max_steps=23, gateway=True)),
        snapshot(
            {
                "type": "claude_cli",
                "config": {
                    "timeout_seconds": None,
                    "model_name": "Claude CLI",
                    "model": "claude-sonnet-4-6",
                    "gateway": True,
                    "system_prompt": None,
                    "permission_mode": "bypassPermissions",
                    "max_steps": 23,
                    "reasoning_effort": None,
                    "screenshot_encoding": {"mime_type": "image/webp", "quality": 85},
                    "allowed_tools": [
                        "Read",
                        "Write",
                        "Edit",
                        "Bash",
                        "Glob",
                        "Grep",
                        "WebSearch",
                        "WebFetch",
                    ],
                },
            }
        ),
    ),
}


PARENTS = {
    "a uuid ambient trace is the parent": ("3" * 32, str(uuid.UUID("3" * 32))),
    "a non-uuid ambient trace is dropped": ("external-run-id", None),
    "the rollout's own id is not its parent": (str(uuid.UUID(TRACE_ID)), None),
    "the rollout's own id in braces is not its parent": (
        "{" + str(uuid.UUID(TRACE_ID)) + "}",
        None,
    ),
    "the rollout's own id as a urn is not its parent": (f"urn:uuid:{uuid.UUID(TRACE_ID)}", None),
}


@pytest.mark.parametrize(("ambient", "parent"), PARENTS.values(), ids=PARENTS.keys())
async def test_a_hosted_rollout_names_its_parent_trace_only_when_it_is_another_trace(
    ambient: str, parent: str | None, platform: FakeServices
) -> None:
    platform.route("api", "GET", POLL, json={"status": "completed", "reward": 1.0})

    with set_trace_context(ambient):
        await HostedRuntime(poll_interval=0).run(
            ROW, chat_agent(), job_id=JOB_ID, trace_id=TRACE_ID
        )

    assert submitted(platform).get("parent_trace_id") == parent


TERMINAL = {
    "a completed trace": (
        {"status": "completed", "reward": 1.0, "error": None},
        (1.0, "completed", None, {"score": 1.0}, False),
    ),
    "a completed trace with its evaluation": (
        {
            "status": "completed",
            "reward": 0.5,
            "evaluation_result": {"score": 0.5, "content": "half", "info": {"k": 1}},
        },
        (0.5, "completed", None, {"score": 0.5, "content": "half", "info": {"k": 1}}, False),
    ),
    "an errored trace with an errored evaluation": (
        {"status": "error", "evaluation_result": {"score": 0.0, "isError": True}},
        (0.0, "error", None, {"score": 0.0, "isError": True}, True),
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
        (
            0.0,
            "error",
            "hud rpc error -32603: tasks.grade: result must include a numeric 'score'",
            {},
            True,
        ),
    ),
}


@pytest.mark.parametrize(("state", "folded"), TERMINAL.values(), ids=TERMINAL.keys())
async def test_a_terminal_trace_folds_into_the_run_and_its_job(
    state: dict[str, Any], folded: tuple[Any, ...], platform: FakeServices
) -> None:
    platform.route("api", "GET", POLL, json=state)

    job = await Taskset("hosted", [ROW]).run(chat_agent(), runtime=HostedRuntime(poll_interval=0))

    (run,) = job.runs
    assert (run.reward, run.trace.status, run.trace.error, run.grade.raw, run in job.errors) == (
        folded
    )
    assert run.slug == ROW.slug
    body = submitted(platform)
    assert (body["job_id"], body["group_id"]) == (str(uuid.UUID(job.id)), run.group_id)


STALLS = {
    "a trace that never finishes": (
        Reply(json={"status": "queued"}),
        Reply(json={"status": "running"}),
    ),
    "a submission that never returns": (Reply(json={"status": "queued"}, delay=5), None),
}


@pytest.mark.parametrize(("submit", "poll"), STALLS.values(), ids=STALLS.keys())
async def test_a_hosted_rollout_past_its_deadline_is_cancelled_remotely(
    submit: Reply, poll: Reply | None, platform: FakeServices
) -> None:
    platform.route("api", "POST", SUBMIT, submit)
    if poll is not None:
        platform.route("api", "GET", POLL, poll)

    job = await ROW.run(chat_agent(), runtime=HostedRuntime(poll_interval=0), rollout_timeout=0.2)
    await eventually(lambda: platform.requests("api", "POST", CANCEL) != [])

    (run,) = job.runs
    assert (run.trace.status, run.trace.stop_reason, run.trace.error) == (
        "error",
        "timeout",
        IsStr(regex=r"hosted rollout [0-9a-f]{32} did not finish within 0\.2s"),
    )
    assert platform.bodies("api", "POST", CANCEL) == [{"trace_id": str(uuid.UUID(run.trace_id))}]


async def test_the_deprecated_constructor_timeout_is_the_default_deadline(
    platform: FakeServices,
) -> None:
    platform.route("api", "GET", POLL, json={"status": "running"})
    with pytest.warns(DeprecationWarning, match="rollout_timeout=... to Task.run"):
        runtime = HostedRuntime(poll_interval=0, run_timeout=0.2)

    job = await ROW.run(chat_agent(), runtime=runtime)

    assert job.runs[0].trace.stop_reason == "timeout"
    await eventually(lambda: platform.requests("api", "POST", CANCEL) != [])


async def test_cancelling_a_hosted_rollout_cancels_it_remotely(platform: FakeServices) -> None:
    platform.route("api", "GET", POLL, json={"status": "running"})
    pending = asyncio.create_task(ROW.run(chat_agent(), runtime=HostedRuntime(poll_interval=0)))
    await eventually(lambda: platform.requests("api", "GET", POLL) != [])

    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    await eventually(lambda: platform.requests("api", "POST", CANCEL) != [])

    trace_id = submitted(platform)["trace_id"]
    assert platform.bodies("api", "POST", CANCEL) == [{"trace_id": trace_id}]


REFUSALS: dict[str, tuple[Agent, dict[str, str | None], str]] = {
    "an agent the platform cannot run": (
        ScriptedAgent("3"),
        {"HUD_API_KEY": "k"},
        "hosted execution supports the registered agent types (claude, claude_cli, codex_cli, "
        "openai, gemini, openai_compatible); got ScriptedAgent",
    ),
    "no API key": (chat_agent(), {"HUD_API_KEY": None}, "HUD_API_KEY is required"),
}


@pytest.mark.parametrize(("agent", "variables", "error"), REFUSALS.values(), ids=REFUSALS.keys())
async def test_a_hosted_rollout_that_cannot_be_submitted_is_a_failed_run(
    agent: Agent,
    variables: dict[str, str | None],
    error: str,
    services: FakeServices,
    hud_env: HudEnv,
) -> None:
    hud_env.set(**variables)

    job = await ROW.run(agent, runtime=HostedRuntime(poll_interval=0))

    (run,) = job.runs
    assert (run.trace.status, run.trace.error, run in job.errors) == ("error", error, True)
    assert services.requests() == []


async def test_a_rejected_submission_is_a_failed_run(platform: FakeServices) -> None:
    platform.route("api", "POST", SUBMIT, json={"detail": "unknown env 'sums'"}, status=422)

    job = await ROW.run(chat_agent(), runtime=HostedRuntime(poll_interval=0))

    (run,) = job.runs
    assert run.trace.status == "error"
    assert "unknown env 'sums'" in str(run.trace.error)
    assert platform.requests("api", "GET", POLL) == []
