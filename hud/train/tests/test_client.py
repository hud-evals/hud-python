from __future__ import annotations

import logging
from typing import Any

import pytest

from hud.eval.run import Run
from hud.train import TrainingClient
from hud.utils.gateway import list_gateway_models

_MODEL_ID = "00000000-0000-0000-0000-000000000001"


async def test_training_rejects_timed_out_run_without_explicit_reward() -> None:
    run = Run.failed("rollout timed out")
    run.trace.stop_reason = "timeout"

    with pytest.raises(ValueError, match="explicit training reward"):
        await TrainingClient("test-model").forward_backward([run], group_size=1)


async def test_training_rejects_incomplete_run_groups() -> None:
    runs = [Run(None, "", {}) for _ in range(4)]
    for index, run in enumerate(runs):
        run.trace.trace_id = str(index)
        run.group_id = "a" if index < 3 else "b"

    with pytest.raises(ValueError, match="incomplete GRPO groups"):
        await TrainingClient("test-model").forward_backward(runs, group_size=2)


async def test_slug_resolves_through_the_clients_own_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("hud.settings.settings.api_key", None)
    list_gateway_models.cache_clear()
    seen: list[tuple[str, str | None]] = []

    def catalog(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        seen.append((url, kwargs.get("api_key")))
        return {
            "items": [{"id": _MODEL_ID, "model_name": "owner/model"}],
            "total": 1,
        }

    monkeypatch.setattr("hud.utils.platform.make_request_sync", catalog)
    client = TrainingClient("owner/model", api_key="team-key", api_url="https://api.example")
    try:
        url = await client._train_url("optim_step")
    finally:
        list_gateway_models.cache_clear()
    assert url.startswith(f"{client._base_url}/v1/models/{_MODEL_ID}/")
    assert seen and seen[0][0].startswith("https://api.example/v2/models")
    assert seen[0][1] == "team-key"


def _grpo_runs(groups: list[str], rewards: list[float]) -> list[Run]:
    runs = [Run(None, "", {}) for _ in rewards]
    for index, (run, group_id, reward) in enumerate(zip(runs, groups, rewards, strict=True)):
        run.trace.trace_id = str(index)
        run.group_id = group_id
        run.grade.reward = reward
    return runs


async def _forward_backward(runs: list[Run], monkeypatch: pytest.MonkeyPatch) -> None:
    """Run forward_backward against a stubbed training service (a UUID model id
    resolves without a catalog lookup)."""

    async def fake_request(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        return {"metrics": {}, "num_datums": len(runs)}

    monkeypatch.setattr("hud.train.base.make_request", fake_request)
    await TrainingClient(_MODEL_ID).forward_backward(runs, group_size=2)


async def test_training_warns_when_no_group_has_reward_spread(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Identical rewards within every group zero every GRPO advantage, so the
    # pass accumulates nothing and the later optim_step fails confusingly.
    runs = _grpo_runs(["a", "a", "b", "b"], [1.0, 1.0, 0.0, 0.0])

    with caplog.at_level(logging.WARNING, logger="hud.train.client"):
        await _forward_backward(runs, monkeypatch)

    assert "no gradient" in caplog.text


async def test_training_is_quiet_when_a_group_has_reward_spread(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = _grpo_runs(["a", "a", "b", "b"], [0.0, 1.0, 0.0, 1.0])

    with caplog.at_level(logging.WARNING, logger="hud.train.client"):
        await _forward_backward(runs, monkeypatch)

    assert "no gradient" not in caplog.text


@pytest.fixture
async def grouped_runs(monkeypatch: pytest.MonkeyPatch) -> list[Run]:
    from hud import Environment, Taskset
    from hud.agents.base import Agent
    from hud.agents.types import AgentStep, Sample
    from hud.settings import settings

    monkeypatch.setattr(settings, "telemetry_enabled", False)
    monkeypatch.setattr(settings, "api_key", None)
    env = Environment("training-groups")

    @env.template()
    async def task(name: str, reward: float):
        yield name
        yield reward

    class FixedAgent(Agent):
        async def __call__(self, run: Run) -> None:
            run.record(
                AgentStep(
                    sample=Sample(
                        prompt_token_ids=[10], output_token_ids=[20], output_logprobs=[-0.1]
                    )
                )
            )
            run.trace.content = "answer"

    job = await Taskset("groups", [task(name="A", reward=1.0), task(name="B", reward=0.0)]).run(
        FixedAgent(), group=2
    )
    assert not job.errors
    return job.runs


@pytest.mark.parametrize("method", ["forward", "forward_backward"])
async def test_training_rejects_interleaved_groups_before_http(
    grouped_runs: list[Run], method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def request(*args: Any, **kwargs: Any) -> dict[str, Any]:
        pytest.fail("interleaved groups must not reach the training service")

    monkeypatch.setattr("hud.train.base.make_request", request)
    interleaved = [grouped_runs[index] for index in (0, 2, 1, 3)]
    with pytest.raises(ValueError, match="groups must be contiguous"):
        await getattr(TrainingClient(_MODEL_ID), method)(interleaved, group_size=2)


@pytest.mark.parametrize("method", ["forward", "forward_backward"])
async def test_training_keeps_contiguous_groups_in_order(
    grouped_runs: list[Run], method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def request(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        inputs = kwargs["json"]["inputs"]
        assert [item["reward"] for item in inputs] == [1.0, 1.0, 0.0, 0.0]
        assert [item["trace_id"] for item in inputs] == [run.trace_id for run in grouped_runs]
        if url.endswith("/forward"):
            return {"forward_id": "offline", "data": []}
        return {"metrics": {}, "num_datums": len(inputs)}

    monkeypatch.setattr("hud.train.base.make_request", request)
    await getattr(TrainingClient(_MODEL_ID), method)(grouped_runs, group_size=2)


@pytest.mark.parametrize("method", ["forward", "forward_backward"])
async def test_training_default_group_checks_the_whole_batch(
    grouped_runs: list[Run],
    method: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def request(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        assert "group_size" not in kwargs["json"]
        if url.endswith("/forward"):
            return {"forward_id": "offline", "data": []}
        return {"metrics": {}, "num_datums": len(grouped_runs)}

    monkeypatch.setattr("hud.train.base.make_request", request)
    with caplog.at_level(logging.WARNING, logger="hud.train.client"):
        await getattr(TrainingClient(_MODEL_ID), method)(grouped_runs)
    assert "no gradient" not in caplog.text
