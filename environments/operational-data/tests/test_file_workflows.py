"""File-only examples expose their own forms and use the shared workspace lifecycle."""

import importlib
import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import ModuleType

import httpx
import pytest
from hud import Taskset
from hud.environment import workspace as workspace_module
from hud.environment.server import TaskRunner
from hud.graders import SubScore
from openpyxl import Workbook

import agent_workspace
from data_sources import staging

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class FileWorkflow:
    module: str
    environment: str
    template: str
    groups: tuple[str, ...]
    slug: str


CASES = (
    FileWorkflow(
        "env",
        "operational-data",
        "review",
        ("contracts", "invoices", "payments", "credits"),
        "supplier-settlement-review",
    ),
    FileWorkflow(
        "env",
        "operational-data",
        "plan",
        ("inventory", "policy", "briefing", "emails"),
        "launch-dispatch-plan",
    ),
)


@pytest.fixture(params=CASES, ids=lambda case: case.environment)
def case(request: pytest.FixtureRequest) -> FileWorkflow:
    return request.param


@pytest.fixture
def workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    monkeypatch.setattr(agent_workspace, "WORKSPACE_ROOT", root)
    return root


@pytest.fixture
def file_template(
    case: FileWorkflow, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> ModuleType:
    monkeypatch.setattr(
        workspace_module,
        "usable_bwrap",
        lambda: workspace_module.Bubblewrap("/usr/bin/bwrap"),
    )
    monkeypatch.setattr(staging.settings, "file_tracking_enabled", False)
    return importlib.import_module(case.module)


def test_file_workflow_has_business_specific_upload_fields(
    case: FileWorkflow, file_template: ModuleType
):
    assert file_template.env.name == case.environment
    assert set(file_template.env.templates) == {"assist", "review", "plan"}
    schema = file_template.env.templates[case.template].manifest_entry()["args"]
    properties = schema["properties"]
    assert set(properties) == {
        "prompt",
        "bundled",
        "deliverables",
        "criteria",
        "as_of",
        "hud_api_key",
        *case.groups,
    }
    assert set(schema["required"]) == {"prompt", "criteria"}
    assert properties["prompt"]["x-hud-hint"] == "prompt"
    assert properties["criteria"]["x-hud-hint"] == "grading"
    for name in case.groups:
        assert properties[name]["type"] == "array"
        assert properties[name]["x-hud-hint"] == "data-files"
        assert properties[name]["items"]["$ref"] == "#/$defs/DataFileRef"
    bundled = schema["$defs"]["BundledAttachments"]
    assert set(bundled["properties"]) == {"attachments"}
    serialized = json.dumps(schema).lower()
    for forbidden in ("slack", "jira", "calendar", "x-hud-data-source"):
        assert forbidden not in serialized


async def test_file_workflow_stages_uploads_and_grades_saved_deliverables(
    case: FileWorkflow,
    file_template: ModuleType,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    content = {f"{group}-upload": f"Evidence for {group}.\n".encode() for group in case.groups}
    requested: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "storage.test":
            assert "Authorization" not in request.headers
            return httpx.Response(200, content=content[request.url.path.lstrip("/")])
        assert request.headers["Authorization"] == "Bearer fixture-runtime-key"
        parts = request.url.path.split("/")
        file_id = parts[3]
        requested.append(file_id)
        if parts[-1] == "download":
            return httpx.Response(200, json={"url": f"https://storage.test/{file_id}"})
        return httpx.Response(200, json={"filename": f"{file_id}.txt"})

    monkeypatch.setattr(staging.settings, "api_key", "fixture-runtime-key")
    monkeypatch.setattr(staging.settings, "hud_api_url", "https://api.test")
    monkeypatch.setattr(
        staging.httpx,
        "AsyncClient",
        partial(httpx.AsyncClient, transport=httpx.MockTransport(handler)),
    )
    judged: dict[str, object] = {}

    async def judge(**kwargs: object) -> SubScore:
        judged.update(kwargs)
        return SubScore(
            name="LLMJudgeGrader",
            value=1.0,
            children=[
                SubScore(name=requirement, weight=weight, value=1.0)
                for requirement, weight in kwargs["criteria"]
            ],
        )

    workflow = importlib.import_module("workflow")
    monkeypatch.setattr(workflow.LLMJudgeGrader, "compute_score", judge)
    prompt = "Review the supplied evidence and save the reconciliation and memo."
    arguments = {
        "prompt": prompt,
        "deliverables": ["reconciliation.xlsx", "memo.md"],
        "criteria": [{"requirement": "Reconciles the evidence.", "weight": 2}],
        "as_of": "2026-05-15",
        **{
            group: [{"file_id": f"{group}-upload", "path": f"{group}/evidence.txt"}]
            for group in case.groups
        },
    }
    runner = TaskRunner(file_template.env.templates[case.template], arguments)
    try:
        frame = await runner.start()
        assert frame["prompt"].startswith("Today is Friday, May 15, 2026.")
        assert prompt in frame["prompt"]
        assert "tools named" not in frame["prompt"]
        assert set(requested) == set(content)
        assert frame["data_files"] == [
            {"path": f"files/{group}/evidence.txt", "file_id": f"{group}-upload"}
            for group in case.groups
        ]
        for group in case.groups:
            path = workspace / "files" / group / "evidence.txt"
            assert path.read_bytes() == content[f"{group}-upload"]
            assert str(path) in frame["prompt"]
            assert f"Evidence for {group}." not in frame["prompt"]
        for name in arguments["deliverables"]:
            assert str(workspace / "outputs" / name) in frame["prompt"]

        workbook = Workbook()
        sheet = workbook.active
        assert sheet is not None
        sheet.append(["Invoice", "Pay"])
        sheet.append(["INV-1", 125])
        sheet.append(["Total", "=SUM(B2:B2)"])
        workbook.save(workspace / "outputs" / "reconciliation.xlsx")
        (workspace / "outputs" / "memo.md").write_text("Supported by the supplied evidence.")

        result = await runner.grade({"answer": "Saved the reconciliation and memo."})
        assert result["score"] == pytest.approx(1.0)
        assert result["info"]["actions"] == []
        assert result["info"]["deliverables"] == [
            {"path": "outputs/reconciliation.xlsx", "status": "saved"},
            {"path": "outputs/memo.md", "status": "saved"},
        ]
        assert judged["question"].startswith(prompt)
        assert judged["model"] == "claude-sonnet-5-5"
        assert judged["max_tokens"] == 16384
        assert judged["criteria"] == [("Reconciles the evidence.", 2.0)]
        assert '["INV-1", 125]' in str(judged["answer"])
        assert '["Total", "[uncached formula: =SUM(B2:B2)]"]' in str(judged["answer"])
        assert "Supported by the supplied evidence." in str(judged["answer"])
    finally:
        await runner.cancel()


async def test_portable_file_examples_bind_and_stage_without_upload_credentials(
    case: FileWorkflow,
    file_template: ModuleType,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(staging.settings, "api_key", None)

    def no_upload_client():
        raise AssertionError("bundled examples must not open a data-file API client")

    workflow = importlib.import_module("workflow")
    monkeypatch.setattr(workflow, "DataFiles", no_upload_client)
    tasks = [task for task in Taskset.from_file(ROOT / "tasks.json") if task.slug == case.slug]
    assert len(tasks) > 0
    for task in tasks:
        assert (task.env, task.id) == (case.environment, case.template)
        assert "file_id" not in json.dumps(task.args)
        assert set(task.args["bundled"]) == {"attachments"}
        native = file_template.env.templates[task.id](**task.args)
        assert (native.env, native.id, native.args) == (task.env, task.id, task.args)
        runner = TaskRunner(file_template.env.templates[task.id], task.args)
        try:
            frame = await runner.start()
            assert frame["data_files"] == []
            assert "tools named" not in frame["prompt"]
            expected = {
                workspace / "files" / Path(relative).name: ROOT / "samples" / relative
                for relative in task.args["bundled"]["attachments"]
            }
            assert len(expected) == len(task.args["bundled"]["attachments"])
            assert {path for path in workspace.rglob("*") if path.is_file()} == set(expected)
            for target, source in expected.items():
                assert target.read_bytes() == source.read_bytes()
                assert str(target) in frame["prompt"]
            for name in task.args["deliverables"]:
                assert str(workspace / "outputs" / name) in frame["prompt"]
        finally:
            await runner.cancel()
