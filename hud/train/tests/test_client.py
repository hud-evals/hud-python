from __future__ import annotations

import logging
from typing import Any

import pytest

from hud.agents.base import Agent
from hud.agents.types import AgentStep, Sample
from hud.environment import Environment
from hud.eval.run import Run
from hud.graders import EvaluationResult
from hud.train import TrainingClient
from hud.train.types import TrajectoryPayload, TrajectorySample
from hud.utils.gateway import list_gateway_models

_MODEL_ID = "00000000-0000-0000-0000-000000000001"


async def test_training_rejects_timed_out_run_without_explicit_reward() -> None:
    run = Run.failed("rollout timed out")
    run.trace.stop_reason = "timeout"

    with pytest.raises(ValueError, match="explicit training reward"):
        await TrainingClient("test-model").forward_backward([run], group_size=1)


class _Answer(Agent):
    """Answers every run; ``tokens`` records a sample as a trainable local model
    does, and ``crash`` fails the run mid-way after answering."""

    def __init__(self, *, tokens: bool, crash: bool = False) -> None:
        super().__init__()
        self._tokens = tokens
        self._crash = crash

    async def __call__(self, run: Any) -> None:
        if self._tokens:
            run.record(
                AgentStep(
                    sample=Sample(
                        prompt_token_ids=[1], output_token_ids=[2], output_logprobs=[-0.1]
                    )
                )
            )
        run.trace.content = "answer"
        if self._crash:
            raise RuntimeError("agent crashed")


async def _graded(
    evaluation: EvaluationResult | float | Exception, *, tokens: bool, crash: bool = False
) -> Run:
    env = Environment("grading")

    @env.template(id="task")
    async def task():
        yield "prompt"
        if isinstance(evaluation, Exception):
            raise evaluation
        yield evaluation

    job = await task().run(_Answer(tokens=tokens, crash=crash))
    return job.runs[0]


def _recording_service(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []

    async def fake_request(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        sent.append(kwargs["json"])
        return {"metrics": {}, "num_datums": len(kwargs["json"]["inputs"])}

    monkeypatch.setattr("hud.train.base.make_request", fake_request)
    return sent


@pytest.mark.parametrize("tokens", [True, False], ids=["inline", "trace_id"])
async def test_training_rejects_run_whose_grading_failed(
    tokens: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The environment reports that grading itself failed. Job.reward excludes the
    # run as having no verdict and its trace still completes, so the reward it
    # carries (here non-zero) must not reach the trainer as a real outcome.
    run = await _graded(
        EvaluationResult(reward=0.7, isError=True, content="hidden tests failed to apply"),
        tokens=tokens,
    )
    assert run.grade.is_error and not run.trace.is_error
    sent = _recording_service(monkeypatch)

    with pytest.raises(ValueError, match="hidden tests failed to apply"):
        await TrainingClient(_MODEL_ID).forward_backward([run])
    assert sent == []


@pytest.mark.parametrize("failure", ["grader_raised", "failed_before_launch"])
async def test_training_rejects_run_without_any_grade(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The trace errored and no grade came back, so the run's 0.0 reward is a
    # default, not a verdict.
    if failure == "grader_raised":
        run = await _graded(RuntimeError("grader crashed"), tokens=True)
    else:
        run = Run.failed("provisioning never finished")
        run.trace.trace_id = "t"
    assert run.trace.is_error and not run.grade.is_error
    sent = _recording_service(monkeypatch)

    with pytest.raises(ValueError, match="no valid grade"):
        await TrainingClient(_MODEL_ID).forward_backward([run])
    assert sent == []


@pytest.mark.parametrize("tokens", [True, False], ids=["inline", "trace_id"])
async def test_training_keeps_valid_grade_after_mid_run_failure(
    tokens: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The agent failed, but the environment still returned a real grade, which
    # Job.reward counts, so training must accept it too.
    run = await _graded(0.7, tokens=tokens, crash=True)
    assert run.trace.is_error and run.grade.raw and not run.grade.is_error
    sent = _recording_service(monkeypatch)

    await TrainingClient(_MODEL_ID).forward_backward([run])

    received = sent[0]["inputs"][0]
    if tokens:
        assert received["reward"] == 0.7
    else:
        assert received == run.trace_id


async def test_training_keeps_genuine_zero_and_explicit_rewards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zero = await _graded(0.0, tokens=True)
    explicit = TrajectoryPayload(
        samples=[TrajectorySample(output_token_ids=[2], output_logprobs=[-0.1])],
        reward=0.25,
        trace_id="t",
    )
    sent = _recording_service(monkeypatch)

    await TrainingClient(_MODEL_ID).forward_backward([zero, explicit])

    assert [item["reward"] for item in sent[0]["inputs"]] == [0.0, 0.25]


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
    _recording_service(monkeypatch)
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


async def test_training_groups_rewards_by_group_id_not_position(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An interleaved batch is valid (groups are keyed by group_id), and its
    # positional slices have spread even when the real groups have none.
    runs = _grpo_runs(["a", "b", "a", "b"], [1.0, 0.0, 1.0, 0.0])

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
