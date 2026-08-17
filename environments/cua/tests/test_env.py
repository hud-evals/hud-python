"""Offline tests for the CUA environment grader composition.

These do NOT touch the virtual desktop (rfb is Linux-only); they drive the @env.template
generator directly and exercise the BashGrader/LLM-judge/fallback paths with deterministic
shell commands. The end-to-end desktop run is a `hud eval ... --runtime hud` rollout.
"""

import env as M

GEN = M.cua_task.func


class TestGrading:
    async def test_empty_fallback_scores_one(self):
        # no bash checks, no criteria -> the desktop_running fallback (reward 1.0)
        gen = GEN(prompt="p")
        await gen.asend(None)
        result = await gen.asend("answer")
        assert result.reward == 1.0
        assert any(s.name == "desktop_running" for s in result.subscores)

    async def test_bash_check_pass_and_fail(self):
        gen = GEN(prompt="p", bash_checks=[{"name": "ok", "command": "true", "weight": 1.0}])
        await gen.asend(None)
        assert (await gen.asend("a")).reward == 1.0

        gen = GEN(prompt="p", bash_checks=[{"name": "no", "command": "false", "weight": 1.0}])
        await gen.asend(None)
        assert (await gen.asend("a")).reward == 0.0

    async def test_no_key_judge_scores_zero_not_error(self, monkeypatch):
        monkeypatch.setattr(M.settings, "api_key", "", raising=False)
        gen = GEN(prompt="p", grading_criteria=["anything"])
        await gen.asend(None)
        result = await gen.asend("answer")
        assert result.reward == 0.0  # llm_judge scores 0 at weight 1.0, no exception
        assert any(s.name == "llm_judge" for s in result.subscores)

    async def test_weights_normalize_across_bash_and_skipped_judge(self, monkeypatch):
        monkeypatch.setattr(M.settings, "api_key", "", raising=False)
        gen = GEN(
            prompt="p",
            bash_checks=[{"name": "ok", "command": "true", "weight": 0.3}],
            grading_criteria=["x"],
        )
        await gen.asend(None)
        result = await gen.asend("answer")
        assert result.reward == 0.5
        weights = {subscore.name: subscore.weight for subscore in result.subscores}
        assert weights == {"ok": 0.5, "llm_judge": 0.5}

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
