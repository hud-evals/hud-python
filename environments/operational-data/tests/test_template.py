"""What the template promises the console and the agent: hinted upload slots, tools, a
workspace, grading."""

from pathlib import Path
from types import TracebackType
from typing import Any, Self

import pytest
from exports import SAMPLES
from fastmcp import Client
from hud.environment import DataFileRef
from hud.graders import SubScore
from openpyxl import Workbook

import agent_workspace
import env as template
import workflow
from data_sources.provider import TaskInProgress
from data_sources.sources import SOURCES
from data_sources.staging import UnsafePath, destination

AGING = "attachments/AP aging 2026-03-31.xlsx"


@pytest.fixture(autouse=True)
def workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    monkeypatch.setattr(agent_workspace, "WORKSPACE_ROOT", root)
    return root


@pytest.fixture
def judged(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    seen: dict[str, object] = {}

    async def fake_compute_score(**kwargs: object) -> SubScore:
        seen.update(kwargs)
        return SubScore(
            name="LLMJudgeGrader",
            value=1.0,
            children=[
                SubScore(name=requirement, weight=weight, value=1.0)
                for requirement, weight in kwargs["criteria"]
            ],
        )

    monkeypatch.setattr(workflow.LLMJudgeGrader, "compute_score", fake_compute_score)
    return seen


def _arguments() -> dict[str, Any]:
    return template.assist.manifest_entry()["args"]


def test_each_data_source_has_exactly_one_upload_slot():
    properties = _arguments()["properties"]
    slots = {
        name: schema["x-hud-data-source"]
        for name, schema in properties.items()
        if "x-hud-data-source" in schema
    }
    assert sorted(slots.values()) == sorted(SOURCES)
    for name, source_id in slots.items():
        assert properties[name]["x-hud-hint"] == "data-files"
        assert properties[name]["items"]["$ref"] == "#/$defs/DataFileRef"
        assert properties[name]["title"] == SOURCES[source_id].title
        assert properties[name]["description"] == SOURCES[source_id].summary


def test_prompt_and_criteria_keep_their_hints():
    arguments = _arguments()
    properties = arguments["properties"]
    assert properties["prompt"]["x-hud-hint"] == "prompt"
    assert properties["criteria"]["x-hud-hint"] == "grading"
    assert set(arguments["required"]) == {"prompt", "criteria"}


def test_attachments_are_a_plain_file_list_and_deliverables_are_names():
    properties = _arguments()["properties"]
    attachments = properties["attachments"]
    assert attachments["x-hud-hint"] == "data-files"
    assert "x-hud-data-source" not in attachments
    assert attachments["items"]["$ref"] == "#/$defs/DataFileRef"
    deliverables = properties["deliverables"]
    assert deliverables["items"] == {"type": "string"}
    assert "x-hud-hint" not in deliverables
    assert "/workspace/outputs" in deliverables["description"]


def test_the_agent_shell_is_sandboxed_as_an_unprivileged_user_with_the_system_python():
    shell = template.env._workspaces["shell"]
    assert shell._require_isolation
    assert not shell.network
    assert (shell._shell_uid, shell._shell_gid) == (1000, 1000)
    assert shell.env["PATH"].split(":")[0] == "/usr/local/bin"


class SampleFiles:
    """Serves samples/ by relative path as file id, in place of the HUD data-file API."""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    async def download(self, ref: DataFileRef, root: Path) -> Path:
        source = SAMPLES / ref.file_id
        target = destination(root, ref.path or source.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as out:
            out.write(source.read_bytes())
        return target


async def test_a_task_serves_its_uploads_as_tools_and_grades_the_changes(
    monkeypatch: pytest.MonkeyPatch, judged: dict[str, object]
):
    monkeypatch.setattr(workflow, "DataFiles", SampleFiles)

    task = template.assist.func(
        prompt="Record the Meridian decision on CLOSE-13.",
        slack=[{"file_id": "slack/Larkspur Slack export.zip"}],
        jira=[{"file_id": "jira/Jira export.csv"}],
        criteria=[{"requirement": "Comments on CLOSE-13 with the short-pay.", "weight": 2}],
        as_of="2026-03-31",
    )
    frame = await task.asend(None)
    assert frame["prompt"].startswith("Today is Tuesday, March 31, 2026.")
    assert "tools named slack_*" in frame["prompt"]
    assert "tools named jira_*" in frame["prompt"]
    assert "calendar_" not in frame["prompt"]

    async with Client(template.tool_server) as agent:
        names = {tool.name for tool in await agent.list_tools()}
        assert "jira_add_comment" in names
        assert not any(name.startswith("calendar_") for name in names)
        await agent.call_tool(
            "jira_add_comment",
            {"key": "CLOSE-13", "body": "Short-paying $44,900 on April 4 unless CM-0219 arrives."},
        )

    result = await task.asend("Commented on CLOSE-13.")
    assert result.reward == pytest.approx(1.0)
    assert judged["criteria"] == [("Comments on CLOSE-13 with the short-pay.", 2.0)]
    assert judged["model"] == "claude-sonnet-5-5"
    assert judged["max_tokens"] == 16384
    assert "claiming the action is not sufficient" in str(judged["answer"])
    assert "jira_add_comment" in str(judged["answer"])
    assert [action["tool"] for action in result.info["actions"]] == ["jira_add_comment"]
    assert result.info["deliverables"] == []

    async with Client(template.tool_server) as agent:
        assert await agent.list_tools() == []


async def test_attachments_land_in_the_workspace_and_deliverables_are_graded(
    monkeypatch: pytest.MonkeyPatch, judged: dict[str, object], workspace: Path
):
    monkeypatch.setattr(workflow, "DataFiles", SampleFiles)
    task = template.assist.func(
        prompt="Build the payment run.",
        slack=[{"file_id": "slack/Larkspur Slack export.zip"}],
        attachments=[{"file_id": AGING}],
        deliverables=["ap-payment-run.xlsx", "memo.md"],
        criteria=[{"requirement": "Short-pays ML-4471."}],
        as_of="2026-03-31",
    )
    frame = await task.asend(None)
    aging = workspace / "files" / "AP aging 2026-03-31.xlsx"
    assert aging.read_bytes() == (SAMPLES / AGING).read_bytes()
    assert frame["data_files"] == [{"path": "files/AP aging 2026-03-31.xlsx", "file_id": AGING}]
    assert f"- {aging}" in frame["prompt"]
    assert f"- {workspace / 'outputs' / 'ap-payment-run.xlsx'}" in frame["prompt"]
    assert "tools named slack_*" in frame["prompt"]

    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.append(["Vendor", "Bill No.", "Amount to pay"])
    sheet.append(["Meridian Logistics", "ML-4471", 44900])
    sheet.append(["Kestrel Partners", "KP-2026-02", 18000])
    sheet.append(["Total", "", "=SUM(C2:C3)"])
    workbook.save(workspace / "outputs" / "ap-payment-run.xlsx")

    result = await task.asend("Saved the run.")
    answer = str(judged["answer"])
    assert '["Meridian Logistics", "ML-4471", 44900]' in answer
    assert '["Total", null, "[uncached formula: =SUM(C2:C3)]"]' in answer
    assert "outputs/memo.md (missing)" in answer
    assert result.info["deliverables"] == [
        {"path": "outputs/ap-payment-run.xlsx", "status": "saved"},
        {"path": "outputs/memo.md", "status": "missing"},
    ]


async def test_every_task_starts_with_an_empty_workspace(
    workspace: Path, judged: dict[str, object]
):
    first = template.assist.func(prompt="Write notes.", criteria=[{"requirement": "Writes."}])
    await first.asend(None)
    (workspace / "outputs" / "notes.md").write_text("left behind")
    (workspace / "scratch").mkdir()
    await first.asend("Done.")

    second = template.assist.func(
        prompt="Write notes.", deliverables=["notes.md"], criteria=[{"requirement": "Writes."}]
    )
    await second.asend(None)
    assert sorted(path.name for path in workspace.iterdir()) == ["files", "outputs"]
    assert list((workspace / "outputs").iterdir()) == []
    result = await second.asend("Done.")
    assert result.info["deliverables"] == [{"path": "outputs/notes.md", "status": "missing"}]


async def test_a_deliverable_outside_the_outputs_folder_is_refused_before_the_task_starts(
    workspace: Path,
):
    task = template.assist.func(
        prompt="Hi.", deliverables=["../env.py"], criteria=[{"requirement": "Greets."}]
    )
    with pytest.raises(UnsafePath):
        await task.asend(None)
    assert not workspace.exists()


async def test_an_overlapping_task_fails_without_touching_the_running_one(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(workflow, "DataFiles", SampleFiles)
    running = template.assist.func(
        prompt="Which close tasks are still open?",
        jira=[{"file_id": "jira/Jira export.csv"}],
        criteria=[{"requirement": "Lists the open tasks."}],
    )
    await running.asend(None)
    try:
        overlapping = template.assist.func(prompt="Hi.", criteria=[{"requirement": "Greets."}])
        with pytest.raises(TaskInProgress):
            await overlapping.asend(None)
        async with Client(template.tool_server) as agent:
            assert "jira_get_issue" in {tool.name for tool in await agent.list_tools()}
    finally:
        await running.aclose()

    async with Client(template.tool_server) as agent:
        assert await agent.list_tools() == []
    following = template.assist.func(prompt="Hi.", criteria=[{"requirement": "Greets."}])
    assert "Today is" in (await following.asend(None))["prompt"]
    await following.aclose()
