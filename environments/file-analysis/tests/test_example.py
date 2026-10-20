"""The selected example exposes authoring controls and grades the saved report."""

from __future__ import annotations

import importlib.util
import json
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize("filename", ["notes.txt", "bank-payments.csv"])
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
            assert (tmp_path / "files" / filename).read_bytes() == (
                ROOT / "fixtures" / "supplier-payments" / filename
            ).read_bytes()
        else:
            frame = await runner.start()
            assert (tmp_path / "files" / filename).read_text() == "Additional context"
            assert len(list((tmp_path / "files").iterdir())) == 8
            assert frame["data_files"] == [{"path": "files/notes.txt", "file_id": "uploaded-record"}]
    finally:
        await runner.cancel()


async def test_missing_report_cannot_pass_from_final_text(monkeypatch, tmp_path, task_args):
    monkeypatch.setattr(example, "WORKSPACE_ROOT", tmp_path)
    runner = TaskRunner(example.reconcile_supplier_payments, task_args)
    await runner.start()
    result = await runner.grade({"answer": "All invoices reconciled correctly."})
    assert result["score"] == 0.0


CASES = json.loads((ROOT / "tasks.json").read_text())
REPORTS = {
    "reconcile_supplier_payments": "reconciliation-report.md",
    "prioritize_support_tickets": "support-priorities.md",
    "synthesize_customer_feedback": "onboarding-feedback.md",
}
INPUTS = {
    "reconcile_supplier_payments": "supplier-payments",
    "prioritize_support_tickets": "support-queue",
    "synthesize_customer_feedback": "onboarding-feedback",
}


def test_runnable_tasks_share_json_data_and_bind_to_the_local_environment(monkeypatch):
    monkeypatch.setattr(example.env, "name", "my-operations-team")
    monkeypatch.setitem(sys.modules, "env", example)
    spec = importlib.util.spec_from_file_location("file_analysis_tasks", ROOT / "tasks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert len(module.tasks) == 3
    assert {task.id for task in module.tasks} == set(REPORTS)
    assert {task.slug for task in module.tasks} == {row["slug"] for row in CASES}
    for task, row in zip(module.tasks, CASES, strict=True):
        assert task.env == example.env.name
        assert task.args == row["args"]
        assert task.agent_config == row["agent_config"]


@pytest.mark.parametrize("row", CASES, ids=lambda row: row["slug"])
async def test_each_task_exposes_controls_and_stages_only_its_input_files_without_a_key(monkeypatch, tmp_path, row):
    monkeypatch.setattr(example, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(example.settings, "api_key", None)
    template = example.env.templates[row["id"]]
    properties = template.manifest_entry()["args"]["properties"]
    assert properties["prompt"]["x-hud-hint"] == "prompt"
    assert properties["attachments"]["x-hud-hint"] == "data-files"
    assert properties["criteria"]["x-hud-hint"] == "grading"
    runner = TaskRunner(template, row["args"])
    try:
        frame = await runner.start()
        expected = ROOT / "fixtures" / INPUTS[row["id"]]
        actual = tmp_path / "files"
        assert {p.name for p in actual.iterdir()} == {p.name for p in expected.iterdir()}
        for path in expected.iterdir():
            assert (actual / path.name).read_bytes() == path.read_bytes()
            assert f"files/{path.name}" in frame["prompt"]
        assert row["args"]["prompt"] in frame["prompt"]
        assert frame["data_files"] == []
        assert not (tmp_path / "tasks.json").exists()
        assert row["args"]["criteria"][0]["requirement"] not in frame["prompt"]
    finally:
        await runner.cancel()


@pytest.mark.parametrize("report_kind", ["missing", "empty", "symlink"])
async def test_invalid_saved_reports_score_zero_without_calling_the_judge(
    monkeypatch, tmp_path, task_args, report_kind
):
    monkeypatch.setattr(example, "WORKSPACE_ROOT", tmp_path)

    async def unexpected_judge(**kwargs):
        pytest.fail("invalid reports must not call the inference gateway")

    monkeypatch.setattr(example.LLMJudgeGrader, "compute_score", unexpected_judge)
    runner = TaskRunner(example.reconcile_supplier_payments, task_args)
    await runner.start()
    report = tmp_path / "reconciliation-report.md"
    if report_kind == "empty":
        report.write_text("   ")
    elif report_kind == "symlink":
        report.symlink_to(ROOT / "tasks.json")
    result = await runner.grade({"answer": "All of the answers, including EUR1137.60, are here."})
    assert result["score"] == 0.0


@pytest.mark.parametrize("row", CASES, ids=lambda row: row["slug"])
@pytest.mark.parametrize("verdict", ["MET", "UNMET"])
async def test_reports_and_original_rubrics_reach_the_judge_gateway(monkeypatch, tmp_path, row, verdict):
    """Exercise real criterion aggregation; only inference responses are mocked."""
    import hud.utils.gateway as gateway

    monkeypatch.setattr(example, "WORKSPACE_ROOT", tmp_path)
    requests = []
    report_text = (
        (ROOT / "tests" / "reports" / f"{row['id']}.md").read_text()
        if verdict == "MET"
        else "All is fine; no evidence or calculations needed."
    )

    async def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps({"criterion_status": verdict, "explanation": "recorded boundary fixture"})
                    )
                )
            ]
        )

    monkeypatch.setattr(
        gateway,
        "build_gateway_client",
        lambda provider: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )
    runner = TaskRunner(example.env.templates[row["id"]], row["args"])
    await runner.start()
    (tmp_path / REPORTS[row["id"]]).write_text(report_text)
    result = await runner.grade({"answer": "This final message must not replace the saved report."})
    assert result["score"] == (1.0 if verdict == "MET" else 0.0)
    assert len(requests) == len(row["args"]["criteria"])
    requirements = [criterion["requirement"] for criterion in row["args"]["criteria"]]
    for request in requests:
        assert request["model"] == "claude-sonnet-5-5"
        message = request["messages"][-1]["content"]
        assert f"<response>\n{report_text}\n</response>" in message
        assert row["args"]["prompt"] in message
        assert sum(requirement in message for requirement in requirements) == 1
    children = result["subscores"][0]["children"]
    assert [child["weight"] for child in children] == [criterion["weight"] for criterion in row["args"]["criteria"]]


async def test_partial_criterion_credit_is_weighted(monkeypatch, tmp_path, task_args):
    import hud.utils.gateway as gateway

    monkeypatch.setattr(example, "WORKSPACE_ROOT", tmp_path)
    credited = task_args["criteria"][0]["requirement"]

    async def create(**kwargs):
        message = kwargs["messages"][-1]["content"]
        verdict = "MET" if credited in message else "UNMET"
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps({"criterion_status": verdict, "explanation": "one supported conclusion"})
                    )
                )
            ]
        )

    monkeypatch.setattr(
        gateway,
        "build_gateway_client",
        lambda provider: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )
    runner = TaskRunner(example.reconcile_supplier_payments, task_args)
    await runner.start()
    (tmp_path / "reconciliation-report.md").write_text(
        "NORTH-1048 has two settled EUR900 payments; EUR900 is recoverable."
    )
    result = await runner.grade({"answer": "partial report saved"})
    assert result["score"] == pytest.approx(0.2)


async def test_environment_refuses_to_start_without_workspace_isolation(monkeypatch):
    from hud.environment import workspace

    monkeypatch.setattr(workspace, "usable_bwrap", lambda: None)
    spec = importlib.util.spec_from_file_location("file_analysis_unisolated", ROOT / "env.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    with pytest.raises(RuntimeError, match="isolation was required"):
        await module.env.start()
