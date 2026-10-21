"""Portable examples stage bundled fixtures through the normal task lifecycle."""

import json
from datetime import date
from functools import partial
from pathlib import Path

import httpx
import pytest
from fastmcp import Client
from hud import Taskset
from hud.environment.server import TaskRunner
from hud.graders import SubScore

import agent_workspace
import env as template
import workflow
from data_sources import staging
from data_sources.provider import TaskInProgress
from data_sources.staging import DataFileError, UnsafePath, copy_bundled

ROOT = Path(__file__).resolve().parent.parent
ROWS = json.loads((ROOT / "tasks.json").read_text())
AGING = "attachments/AP aging 2026-03-31.xlsx"


@pytest.fixture(autouse=True)
def workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    monkeypatch.setattr(agent_workspace, "WORKSPACE_ROOT", root)
    return root


def test_default_tasks_are_the_five_approved_operational_examples():
    assert [row["slug"] for row in ROWS] == [
        "board-review-scheduling",
        "march-close-status",
        "ap-payment-run",
        "supplier-settlement-review",
        "launch-dispatch-plan",
    ]
    assert {(row["env"], row["id"]) for row in ROWS} == {
        ("operational-data", "assist"),
        ("operational-data", "review"),
        ("operational-data", "plan"),
    }


def test_portable_tasks_import_without_upload_ids(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(staging.settings, "api_key", None)
    assert not (ROOT / "uploads.json").exists()
    portable = Taskset.from_file(ROOT / "tasks.json")
    bound = Taskset.from_file(ROOT / "tasks.py")
    assert [task.slug for task in bound] == [task.slug for task in portable]
    assert len(bound) == len(ROWS)
    for task in bound:
        assert task.id == portable.tasks[task.slug].id
        assert task.env == template.env.name
        assert task.columns == portable.tasks[task.slug].columns
        assert task.agent_config == {"model": "claude-sonnet-5-5", "max_steps": 60}
        assert task.args == portable.tasks[task.slug].args
        assert "file_id" not in json.dumps(task.args)


@pytest.mark.parametrize("row", ROWS, ids=[row["slug"] for row in ROWS])
async def test_bundled_examples_start_without_a_data_file_client(
    monkeypatch: pytest.MonkeyPatch, workspace: Path, row: dict
):
    monkeypatch.setattr(staging.settings, "api_key", None)

    def no_upload_client():
        raise AssertionError("bundled examples must not open an authenticated data-file client")

    monkeypatch.setattr(workflow, "DataFiles", no_upload_client)
    runner = TaskRunner(template.env.templates[row["id"]], row["args"])
    try:
        frame = await runner.start()
        today = date.fromisoformat(row["args"]["as_of"])
        assert frame["prompt"].startswith(
            f"Today is {today:%A}, {today:%B} {today.day}, {today.year}."
        )
        assert frame["data_files"] == []
        async with Client(template.tool_server) as agent:
            names = {tool.name for tool in await agent.list_tools()}
        for source in ("slack", "jira", "calendar"):
            assert any(name.startswith(f"{source}_") for name in names) == (
                source in row["args"]["bundled"]
            )
        expected = {
            workspace / "files" / Path(relative).name: ROOT / "samples" / relative
            for relative in row["args"]["bundled"].get("attachments", [])
        }
        assert {path for path in workspace.rglob("*") if path.is_file()} == set(expected)
        for target, source in expected.items():
            assert target.read_bytes() == source.read_bytes()
            assert str(target) in frame["prompt"]
        for name in row["args"].get("deliverables", []):
            assert str(workspace / "outputs" / name) in frame["prompt"]
    finally:
        await runner.cancel()


@pytest.mark.parametrize("following", ["review", "plan"])
async def test_templates_share_one_task_session_and_reset_the_workspace(
    following: str, workspace: Path
):
    arguments = {"prompt": "Inspect the evidence.", "criteria": [{"requirement": "Inspects."}]}
    running = TaskRunner(template.assist, arguments)
    await running.start()
    marker = workspace / "outputs" / "notes.md"
    marker.write_text("The active task's work.")
    overlapping = TaskRunner(template.env.templates[following], arguments)
    try:
        with pytest.raises(TaskInProgress):
            await overlapping.start()
        assert marker.read_text() == "The active task's work."
    finally:
        await overlapping.cancel()
        await running.cancel()

    next_task = TaskRunner(template.env.templates[following], arguments)
    try:
        frame = await next_task.start()
        assert "tools named" not in frame["prompt"]
        assert not marker.exists()
    finally:
        await next_task.cancel()


@pytest.fixture
def upload(monkeypatch: pytest.MonkeyPatch):
    client = httpx.AsyncClient
    monkeypatch.setattr(staging.settings, "api_key", "fixture-runtime-key")
    monkeypatch.setattr(staging.settings, "hud_api_url", "https://api.test")

    def configure(filename: str, data: bytes):
        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "storage.test":
                assert "Authorization" not in request.headers
                return httpx.Response(200, content=data)
            assert request.headers["Authorization"] == "Bearer fixture-runtime-key"
            if request.url.path.endswith("/download"):
                return httpx.Response(200, json={"url": "https://storage.test/blob"})
            return httpx.Response(200, json={"filename": filename})

        monkeypatch.setattr(
            staging.httpx, "AsyncClient", partial(client, transport=httpx.MockTransport(handler))
        )

    return configure


async def test_uploaded_and_bundled_sources_share_the_original_tools_and_grader(
    monkeypatch: pytest.MonkeyPatch, upload, workspace: Path
):
    upload("Jira export.csv", (ROOT / "samples/jira/Jira export.csv").read_bytes())
    judged = {}

    async def judge(**kwargs):
        judged.update(kwargs)
        return SubScore(
            name="LLMJudgeGrader",
            value=1.0,
            children=[
                SubScore(name=requirement, weight=weight, value=1.0)
                for requirement, weight in kwargs["criteria"]
            ],
        )

    monkeypatch.setattr(workflow.LLMJudgeGrader, "compute_score", judge)
    task = template.assist.func(
        prompt="Record the Meridian payment decision.",
        jira=[{"file_id": "uploaded-jira"}],
        bundled={"slack": ["slack/Larkspur Slack export.zip"]},
        criteria=[{"requirement": "Comments on CLOSE-13.", "weight": 2}],
        as_of="2026-03-31",
    )
    await task.asend(None)
    async with Client(template.tool_server) as agent:
        await agent.call_tool(
            "jira_add_comment", {"key": "CLOSE-13", "body": "Pay $44,900 on April 4."}
        )
        names = {tool.name for tool in await agent.list_tools()}
        assert "slack_search_messages" in names
    result = await task.asend("Recorded the payment decision.")
    assert result.reward == pytest.approx(1.0)
    assert [action["tool"] for action in result.info["actions"]] == ["jira_add_comment"]
    assert "Pay $44,900 on April 4." in judged["answer"]
    assert judged["criteria"] == [("Comments on CLOSE-13.", 2.0)]
    assert not any(path.is_file() for path in workspace.rglob("*"))


async def test_bundled_and_uploaded_attachments_cannot_overwrite_each_other(
    upload, workspace: Path
):
    content = b"uploaded attachment fixture"
    upload(Path(AGING).name, content)
    task = template.assist.func(
        prompt="Read the aging report.",
        attachments=[{"file_id": "uploaded-aging"}],
        bundled={"attachments": [AGING]},
        criteria=[{"requirement": "Reads the aging report."}],
    )
    with pytest.raises(DataFileError, match="two data files"):
        await task.asend(None)
    assert (workspace / "files" / Path(AGING).name).read_bytes() == content


async def test_bundled_and_uploaded_source_filename_collisions_are_explicit(upload):
    upload("Jira export.csv", (ROOT / "samples/jira/Jira export.csv").read_bytes())
    task = template.assist.func(
        prompt="Review the close.",
        jira=[{"file_id": "uploaded-jira"}],
        bundled={"jira": ["jira/Jira export.csv"]},
        criteria=[{"requirement": "Reviews the close."}],
    )
    with pytest.raises(DataFileError, match="bundled and uploaded exports share filename"):
        await task.asend(None)


@pytest.mark.parametrize(
    "relative", ["../tasks.json", "/etc/passwd", "slack/../../env.py", "", "slack\\file"]
)
def test_bundled_paths_cannot_leave_the_sample_root(tmp_path: Path, relative: str):
    with pytest.raises(UnsafePath):
        copy_bundled(relative, tmp_path / "staged")


@pytest.mark.parametrize("directory_link", [False, True])
def test_bundled_files_and_directories_cannot_be_symlinks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, directory_link: bool
):
    samples = tmp_path / "samples"
    samples.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "export.json").write_text("fixture")
    if directory_link:
        (samples / "slack").symlink_to(outside, target_is_directory=True)
    else:
        (samples / "slack").mkdir()
        (samples / "slack/export.json").symlink_to(outside / "export.json")
    monkeypatch.setattr(staging, "SAMPLES_ROOT", samples)
    with pytest.raises(UnsafePath, match="symlink"):
        copy_bundled("slack/export.json", tmp_path / "staged")
    assert not (tmp_path / "staged").exists()


async def test_raw_source_exports_cannot_be_selected_as_workspace_attachments(workspace: Path):
    task = template.assist.func(
        prompt="Read the data.",
        bundled={"attachments": ["slack/Larkspur Slack export.zip"]},
        criteria=[{"requirement": "Reads the data."}],
    )
    with pytest.raises(ValueError, match="samples/attachments"):
        await task.asend(None)
    assert not workspace.exists()
