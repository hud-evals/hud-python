"""Offline tests for the CUA environment grader composition.

These do NOT touch the virtual desktop (rfb is Linux-only); they drive the @env.template
generator directly and exercise its configuration and grader composition with deterministic shell
commands. The end-to-end desktop run is a `hud eval ... --runtime hud` rollout.
"""

from unittest.mock import AsyncMock

import pytest

import env as M
import tasks

GEN = M.cua_task.func


@pytest.fixture(autouse=True)
def fresh_substrate(monkeypatch):
    monkeypatch.setattr(M, "_task_started", False)


class TestGrading:
    async def test_empty_fallback_scores_one(self):
        # no bash checks, no criteria -> the desktop_running fallback (reward 1.0)
        gen = GEN(prompt="p")
        await gen.asend(None)
        result = await gen.asend("answer")
        assert result.reward == 1.0
        assert any(s.name == "desktop_running" for s in result.subscores)

    async def test_bash_check_passes(self):
        gen = GEN(prompt="p", bash_checks=[{"name": "ok", "command": "true", "weight": 1.0}])
        await gen.asend(None)
        assert (await gen.asend("a")).reward == 1.0

    async def test_bash_check_fails(self):
        gen = GEN(prompt="p", bash_checks=[{"name": "no", "command": "false", "weight": 1.0}])
        await gen.asend(None)
        assert (await gen.asend("a")).reward == 0.0

    async def test_no_key_judge_fails_before_prompt(self, monkeypatch):
        monkeypatch.setattr(M.settings, "api_key", "", raising=False)
        gen = GEN(prompt="p", grading_criteria=["anything"])
        with pytest.raises(RuntimeError, match="HUD_API_KEY"):
            await gen.asend(None)

    async def test_weights_normalize_across_bash_and_judge(self, monkeypatch):
        monkeypatch.setattr(M.settings, "api_key", "key", raising=False)
        monkeypatch.setattr(
            M.LLMJudgeGrader,
            "grade",
            AsyncMock(return_value=M.SubScore(name="llm_judge", value=1.0, weight=0.5)),
        )
        gen = GEN(
            prompt="p",
            bash_checks=[{"name": "ok", "command": "true", "weight": 0.3}],
            grading_criteria=["x"],
        )
        await gen.asend(None)
        result = await gen.asend("answer")
        assert result.reward == 1.0
        weights = {subscore.name: subscore.weight for subscore in result.subscores}
        assert weights == {"ok": 0.5, "llm_judge": 0.5}

    async def test_second_task_requires_fresh_substrate(self):
        first = GEN(prompt="first")
        await first.asend(None)

        second = GEN(prompt="second")
        with pytest.raises(RuntimeError, match="one task per substrate"):
            await second.asend(None)

        await first.aclose()

    async def test_named_bash_subscores_preserved(self):
        gen = GEN(
            prompt="p",
            bash_checks=[
                {"name": "alpha", "command": "true", "weight": 0.4},
                {"name": "beta", "command": "true", "weight": 0.6},
            ],
        )
        await gen.asend(None)
        result = await gen.asend("a")
        names = {s.name for s in result.subscores}
        assert {"alpha", "beta"} <= names
        assert result.reward == 1.0


def test_create_document_requires_exact_contents():
    command = tasks._create_document.args["bash_checks"][1]["command"]
    assert command == "cmp -s /home/ubuntu/Desktop/hello.txt <(printf 'Hello from HUD!\\n')"
