"""What the example promises: hinted arguments, safe staging, weighted grading."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from hud.graders import SubScore

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import env as env_module  # noqa: E402


def test_every_editable_argument_declares_its_hint():
    """The hints are the point of this environment; a rename must fail here."""
    properties = env_module.review_files.manifest_entry()["args"]["properties"]
    assert properties["prompt"]["x-hud-hint"] == "prompt"
    assert properties["attachments"]["x-hud-hint"] == "data-files"
    assert properties["criteria"]["x-hud-hint"] == "grading"
    assert "hud_api_key" in properties
    required = env_module.review_files.manifest_entry()["args"].get("required") or []
    assert "hud_api_key" not in required
    # The console resolves the reference through the $defs discovery bundles.
    assert properties["attachments"]["items"]["$ref"] == "#/$defs/DataFileRef"
    assert "file_id" in env_module.review_files.manifest_entry()["args"]["$defs"]["DataFileRef"]["properties"]


def test_staging_refuses_a_path_that_leaves_the_files_directory(tmp_path: Path):
    """An uploaded file names its own destination, so the path is untrusted."""
    assert env_module._destination(tmp_path, "notes/summary.md") == tmp_path / "notes/summary.md"
    for hostile in ("../escape.md", "/etc/passwd", "..\\escape.md"):
        with pytest.raises(env_module.DataFileError):
            env_module._destination(tmp_path, hostile)


async def test_criteria_reach_the_judge_unchanged(monkeypatch: pytest.MonkeyPatch):
    """The argument is the grader's input, so nothing may be rewritten en route."""
    seen: dict[str, object] = {}

    async def fake_compute_score(**kwargs: object):
        seen.update(kwargs)
        return SubScore(name="LLMJudgeGrader", value=1.0)

    monkeypatch.setattr(env_module.LLMJudgeGrader, "compute_score", fake_compute_score)
    criteria = [
        env_module.Criterion(requirement="Recommends an interview.", weight=3),
        env_module.Criterion(requirement="Invents an employer.", weight=-2),
    ]

    result = await env_module._grade("Hire her.", "Should we interview her?", criteria)

    assert seen["criteria"] == [("Recommends an interview.", 3.0), ("Invents an employer.", -2.0)]
    assert seen["answer"] == "Hire her."
    assert seen["question"] == "Should we interview her?"
    assert result.reward == pytest.approx(1.0)


async def test_no_criteria_scores_zero_without_raising():
    """An empty list is a misconfigured task, not a crashed rollout."""
    result = await env_module._grade("Hire her.", "", [])
    assert result.reward == pytest.approx(0.0)


async def test_staging_directory_is_cleared_between_tasks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """A prior task's leftovers (including symlinks) must not survive into staging."""

    async def fake_compute_score(**kwargs: object):
        return SubScore(name="LLMJudgeGrader", value=1.0)

    async def fake_stage(refs, root: Path, hud_api_key=None):
        return []

    monkeypatch.setattr(env_module.LLMJudgeGrader, "compute_score", fake_compute_score)
    monkeypatch.setattr(env_module, "_stage", fake_stage)
    monkeypatch.setattr(env_module, "WORKSPACE_ROOT", tmp_path)

    files_dir = tmp_path / env_module.FILES_DIRNAME
    files_dir.mkdir(parents=True)
    (files_dir / "escape").symlink_to(tmp_path / "outside")

    task = env_module.review_files.func(prompt="p", attachments=[], criteria=[])
    await task.asend(None)

    assert files_dir.is_dir()
    assert list(files_dir.iterdir()) == []


async def test_arguments_arrive_as_plain_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """A task run sends JSON, not model instances, so the template must validate."""
    seen: dict[str, object] = {}

    async def fake_compute_score(**kwargs: object):
        seen.update(kwargs)
        return SubScore(name="LLMJudgeGrader", value=1.0)

    staged: list[env_module.DataFileRef] = []

    async def fake_stage(
        refs: list[env_module.DataFileRef],
        root: Path,
        hud_api_key: str | None = None,
    ):
        staged.extend(refs)
        return [{"path": "files/resume.pdf", "file_id": refs[0].file_id}]

    monkeypatch.setattr(env_module.LLMJudgeGrader, "compute_score", fake_compute_score)
    monkeypatch.setattr(env_module, "_stage", fake_stage)
    monkeypatch.setattr(env_module, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(env_module.settings, "api_key", None)

    task = env_module.review_files.func(
        prompt="Summarise the role.",
        attachments=[{"file_id": "8b1f", "path": "resume.pdf"}],
        criteria=[{"requirement": "Recommends an interview.", "weight": 2}],
        hud_api_key="test-key",
    )
    frame = await task.asend(None)

    # Each argument reached its model rather than staying a bare dict.
    assert staged == [env_module.DataFileRef(file_id="8b1f", path="resume.pdf")]
    assert "Summarise the role." in frame["prompt"]
    assert frame["data_files"] == [{"path": "files/resume.pdf", "file_id": "8b1f"}]

    # A task run sends the agent's final text, the same as a real grade call.
    result = await task.asend("Hire her.")

    assert seen["criteria"] == [("Recommends an interview.", 2.0)]
    assert result.reward == pytest.approx(1.0)
