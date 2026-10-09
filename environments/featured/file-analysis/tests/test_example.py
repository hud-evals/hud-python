"""The selected example exposes authoring controls and grades the saved report."""

from __future__ import annotations

import importlib.util
import json
import sys
from functools import partial
from pathlib import Path

import httpx
import pytest
from hud.environment.server import TaskRunner
from hud.graders import SubScore

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("file_analysis_env", ROOT / "env.py")
assert SPEC is not None and SPEC.loader is not None
example = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = example
SPEC.loader.exec_module(example)


@pytest.fixture
def task_args():
    return json.loads((ROOT / "tasks.json").read_text())[0]["args"]


def test_selected_example_exposes_authoring_controls(task_args):
    schema = example.reconcile_supplier_payments.manifest_entry()["args"]
    for field, hint in (("prompt", "prompt"), ("attachments", "data-files"), ("criteria", "grading")):
        assert schema["properties"][field]["x-hud-hint"] == hint
        assert field in task_args
    assert task_args["attachments"] == []


async def test_edited_prompt_and_rubric_grade_the_saved_report(monkeypatch, tmp_path, task_args):
    monkeypatch.setattr(example, "WORKSPACE_ROOT", tmp_path)
    seen = {}

    async def judge(**kwargs):
        seen.update(kwargs)
        return SubScore(name="judge", value=1.0)

    monkeypatch.setattr(example.LLMJudgeGrader, "compute_score", judge)
    task_args.update(
        prompt="Identify the payment records.", criteria=[{"requirement": "Names the records.", "weight": 3}]
    )
    runner = TaskRunner(example.reconcile_supplier_payments, task_args)
    frame = await runner.start()
    assert "Identify the payment records." in frame["prompt"]
    assert len(list((tmp_path / "files").iterdir())) == 7
    (tmp_path / "reconciliation-report.md").write_text("The report from the workspace.")
    result = await runner.grade({"answer": "This final message is not the report."})
    assert result["score"] == 1.0
    assert seen["answer"] == "The report from the workspace."
    assert seen["question"] == task_args["prompt"]
    assert seen["criteria"] == [("Names the records.", 3.0)]
    assert seen["model"] == "claude-sonnet-5-5"


@pytest.mark.parametrize("filename", ["extra-notes.md", "bank-payments.csv"])
async def test_uploads_supplement_but_cannot_replace_sample_records(monkeypatch, tmp_path, task_args, filename):
    monkeypatch.setattr(example, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(example.settings, "api_key", "test-runtime-key")
    monkeypatch.setattr(example.settings, "hud_api_url", "https://api.test")

    def respond(request):
        if request.url.host == "storage.test":
            assert "Authorization" not in request.headers
            return httpx.Response(200, content=b"Additional context")
        assert request.headers["Authorization"] == "Bearer test-runtime-key"
        if request.url.path.endswith("/download"):
            return httpx.Response(200, json={"url": "https://storage.test/document"})
        return httpx.Response(200, json={"filename": filename})

    monkeypatch.setattr(
        example.httpx, "AsyncClient", partial(httpx.AsyncClient, transport=httpx.MockTransport(respond))
    )
    task_args["attachments"] = [{"file_id": "uploaded-record"}]
    runner = TaskRunner(example.reconcile_supplier_payments, task_args)
    try:
        if filename == "bank-payments.csv":
            with pytest.raises(FileExistsError):
                await runner.start()
            assert (tmp_path / "files" / filename).read_bytes() == (ROOT / "fixtures" / filename).read_bytes()
        else:
            frame = await runner.start()
            assert (tmp_path / "files" / filename).read_text() == "Additional context"
            assert len(list((tmp_path / "files").iterdir())) == 8
            assert frame["data_files"] == [{"path": "files/extra-notes.md", "file_id": "uploaded-record"}]
    finally:
        await runner.cancel()


async def test_missing_report_cannot_pass_from_final_text(monkeypatch, tmp_path, task_args):
    monkeypatch.setattr(example, "WORKSPACE_ROOT", tmp_path)
    runner = TaskRunner(example.reconcile_supplier_payments, task_args)
    await runner.start()
    result = await runner.grade({"answer": "All invoices reconciled correctly."})
    assert result["score"] == 0.0
