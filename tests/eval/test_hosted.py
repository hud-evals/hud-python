"""``HostedRuntime``: the platform runs the whole rollout; this process submits it, polls
its trace to a terminal state, and folds the result into a ``Run``. A deadline or a
cancel here cancels the rollout on the platform."""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any

import pytest

from hud.agents import OpenAIChatAgent
from hud.agents.types import OpenAIChatConfig
from hud.eval import HostedRuntime, Task, Taskset
from hud.types import AgentType
from tests.conftest import eventually
from tests.fixtures.services import Reply

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import HudEnv
    from tests.fixtures.services import FakeServices

SUBMIT = "/v2/rollouts/submit"
POLL = "/v2/trace/{id}"
CANCEL = "/v2/rollouts/cancel"


@pytest.fixture(autouse=True)
def signed_in(hud_env: HudEnv) -> None:
    hud_env.set(HUD_API_KEY="k")


def chat_agent() -> OpenAIChatAgent:
    return OpenAIChatAgent(
        OpenAIChatConfig(model="test-model", api_key="secret", base_url="http://model.test")
    )


async def test_a_hosted_rollout_submits_the_row_and_its_agent_then_polls_to_its_end(
    services: FakeServices, tmp_path: Path
) -> None:
    (tmp_path / "compose.yaml").write_text("services:\n  main:\n    image: sums:latest\n")
    services.route("api", "POST", SUBMIT, json={"status": "queued"})
    services.route(
        "api",
        "GET",
        POLL,
        Reply(json={"status": "pending"}),
        Reply(json={"status": "running"}),
        Reply(json={"status": "completed", "reward": 0.5}),
    )
    row = Task.model_validate(
        {
            "env": "sums",
            "id": "add",
            "args": {"a": 1, "b": 2},
            "agent_config": {"timeout_seconds": 45.0},
            "runtime_config": {"compose": {"document": "compose.yaml"}},
            "verifier": {"env": "judge", "id": "verify", "runtime_config": {"image": "judge:1"}},
        },
        context={"base_path": tmp_path},
    )

    job = await Taskset("sums", [row]).run(chat_agent(), runtime=HostedRuntime(poll_interval=0))

    (run,) = job.runs
    (submitted,) = services.requests("api", "POST", SUBMIT)
    body = submitted.json
    assert (body["trace_id"], body["job_id"], body["group_id"]) == (
        str(uuid.UUID(run.trace_id)),
        str(uuid.UUID(job.id)),
        run.group_id,
    )
    assert (body["env"], body["task"], body["args"]) == ("sums", "add", {"a": 1, "b": 2})
    assert (body["agent"]["type"], body["agent"]["config"]["timeout_seconds"]) == (
        "openai_compatible",
        45.0,
    )
    assert body["runtime_config"]["compose"]["document"]["services"]["main"]["image"] == (
        "sums:latest"
    )
    assert (body["verifier"]["env"], body["verifier"]["runtime_config"]) == (
        "judge",
        {"image": "judge:1"},
    )
    assert "secret" not in str(body)
    assert (run.reward, run.trace.status, run.runtime) == (
        0.5,
        "completed",
        f"hud://trace/{run.trace_id}",
    )
    assert services.requests("api", "POST", "/v2/trace/{id}/exit") == []


@pytest.mark.parametrize(
    ("state", "reward", "status", "error", "graded"),
    [
        pytest.param(
            {"status": "completed", "reward": 0.5, "evaluation_result": {"score": 0.5}},
            0.5,
            "completed",
            None,
            True,
            id="completed-with-its-evaluation",
        ),
        pytest.param(
            {"status": "error", "reward": None, "error": "env exploded"},
            0.0,
            "error",
            "env exploded",
            False,
            id="errored-without-a-reward",
        ),
        pytest.param(
            {"status": "error", "reward": 0.75, "error": "agent exploded"},
            0.75,
            "error",
            "agent exploded",
            True,
            id="errored-but-graded",
        ),
        pytest.param(
            {"status": "cancelled", "reward": None},
            0.0,
            "cancelled",
            None,
            False,
            id="cancelled-before-grading",
        ),
        pytest.param(
            {"status": "completed"},
            0.0,
            "completed",
            None,
            False,
            id="completed-without-a-reward",
        ),
        pytest.param(
            {"status": "completed", "evaluation_result": {"score": "high"}},
            0.0,
            "error",
            "'score'",
            False,
            id="an-evaluation-without-a-numeric-score",
        ),
    ],
)
async def test_the_trace_the_platform_ends_with_is_the_runs_outcome(
    state: dict[str, Any],
    reward: float,
    status: str,
    error: str | None,
    graded: bool,
    services: FakeServices,
) -> None:
    services.route("api", "POST", SUBMIT, json={"status": "queued"})
    services.route("api", "GET", POLL, json=state)

    job = await Task(env="sums", id="add").run(chat_agent(), runtime=HostedRuntime(poll_interval=0))

    (run,) = job.runs
    assert (run.reward, run.trace.status, run in job.errors) == (reward, status, not graded)
    if error is None:
        assert run.trace.error is None
    else:
        assert error in (run.trace.error or "")


@pytest.mark.parametrize(
    "submit_delay", [0, 5], ids=["while-the-platform-runs-it", "before-the-platform-accepts-it"]
)
async def test_a_hosted_rollout_past_its_deadline_is_cancelled_on_the_platform(
    submit_delay: float, services: FakeServices
) -> None:
    services.route("api", "POST", SUBMIT, Reply(json={"status": "queued"}, delay=submit_delay))
    services.route("api", "GET", POLL, json={"status": "running"})
    services.route("api", "POST", CANCEL, json={"status": "accepted"})

    job = await Task(env="sums", id="add").run(
        chat_agent(), runtime=HostedRuntime(poll_interval=0), rollout_timeout=1
    )

    (run,) = job.runs
    assert (run.trace.status, run.trace.stop_reason) == ("error", "timeout")
    await eventually(lambda: len(services.requests("api", "POST", CANCEL)) == 1, within=10)
    assert services.bodies("api", "POST", CANCEL) == [{"trace_id": str(uuid.UUID(run.trace_id))}]


async def test_cancelling_a_hosted_rollout_cancels_it_on_the_platform(
    services: FakeServices,
) -> None:
    services.route("api", "POST", SUBMIT, json={"status": "queued"})
    services.route("api", "GET", POLL, json={"status": "running"})
    services.route("api", "POST", CANCEL, json={"status": "accepted"})

    pending = asyncio.create_task(
        Task(env="sums", id="add").run(chat_agent(), runtime=HostedRuntime(poll_interval=0))
    )
    await eventually(lambda: services.requests("api", "POST", SUBMIT) != [])
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    await eventually(lambda: len(services.requests("api", "POST", CANCEL)) == 1, within=10)


@pytest.mark.parametrize(
    ("agent_type", "config"),
    [
        ("claude", {"model": "claude-sonnet-4-6", "max_steps": 5, "tool_timeout_seconds": 30}),
        ("openai", {"model": "gpt-5.6", "prompt_cache_key": "shared"}),
        ("gemini", {"model": "gemini-m", "max_output_tokens": 512}),
        ("openai_compatible", {"model": "m", "completion_kwargs": {"temperature": 0.2}}),
        ("claude_cli", {"reasoning_effort": "max", "max_steps": 3}),
        ("codex_cli", {"sandbox": "read-only", "reasoning_effort": "high"}),
    ],
)
async def test_the_submission_carries_the_agents_type_and_the_users_config(
    agent_type: str, config: dict[str, Any], services: FakeServices
) -> None:
    services.route("api", "POST", SUBMIT, json={"status": "queued"})
    services.route("api", "GET", POLL, json={"status": "completed", "reward": 1.0})
    kind = AgentType(agent_type).cls
    agent = kind(kind.config_cls.model_validate({**config, "gateway": True}))

    await Task(env="sums", id="add").run(agent, runtime=HostedRuntime(poll_interval=0))

    (body,) = services.bodies("api", "POST", SUBMIT)
    assert body["agent"]["type"] == agent_type
    assert config.items() <= body["agent"]["config"].items()
