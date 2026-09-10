"""What the example promises: hinted arguments, safe staging, weighted grading."""

from __future__ import annotations

import asyncio
import json
import sys
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Barrier, Thread

import httpx
import pytest
from hud import LocalRuntime, connect
from hud.clients import HudProtocolError
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

    async def fake_stage(refs, root: Path):
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


@pytest.mark.parametrize("fail_second", [False, True])
async def test_concurrent_sessions_use_the_runtime_credential(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fail_second: bool
):
    monkeypatch.setattr(env_module, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(env_module.settings, "api_key", "process-key")
    arrived = Barrier(2, timeout=5)
    seen = {}

    class Gateway(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            prompt = body["messages"][-1]["content"]
            answer = prompt.split("<response>\n", 1)[1].split("\n</response>", 1)[0]
            seen[answer] = self.headers["Authorization"]
            arrived.wait()
            assert env_module.settings.api_key == "process-key"
            rejected = fail_second and answer == "second"
            payload = json.dumps(
                {"error": {"message": "judge rejected request"}}
                if rejected
                else {
                    "choices": [
                        {
                            "message": {
                                "content": '{"criterion_status":"MET","explanation":"fixture"}',
                            }
                        }
                    ]
                }
            ).encode()
            self.send_response(400 if rejected else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    first_task = env_module.review_files(
        prompt="Give an answer.",
        attachments=[],
        criteria=[{"requirement": "An answer is present.", "weight": 1}],
        hud_api_key="first-key",
    )
    second_task = first_task.model_copy(update={"args": {**first_task.args, "hud_api_key": "second-key"}})
    with ThreadingHTTPServer(("127.0.0.1", 0), Gateway) as server:
        monkeypatch.setattr(env_module.settings, "hud_gateway_url", f"http://127.0.0.1:{server.server_port}")
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            async with (
                LocalRuntime(env_module.env)(first_task) as runtime,
                connect(runtime) as first,
                connect(runtime) as second,
            ):
                await first.start_task(first_task.id, first_task.args)
                await second.start_task(second_task.id, second_task.args)
                results = await asyncio.gather(
                    first.grade({"answer": "first"}),
                    second.grade({"answer": "second"}),
                    return_exceptions=True,
                )
        finally:
            server.shutdown()
            thread.join()

    assert seen == {"first": "Bearer process-key", "second": "Bearer process-key"}
    assert env_module.settings.api_key == "process-key"
    for index, result in enumerate(results):
        if fail_second and index == 1:
            assert isinstance(result, HudProtocolError)
            assert "judge rejected request" in str(result)
            continue
        assert isinstance(result, dict)
        assert result["score"] == 1.0
        assert "first-key" not in json.dumps(result)
        assert "second-key" not in json.dumps(result)


async def test_file_staging_uses_the_runtime_credential(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(env_module, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(env_module.settings, "api_key", "runtime-key")
    monkeypatch.setattr(env_module.settings, "hud_api_url", "https://api.test")

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "storage.test":
            assert "Authorization" not in request.headers
            return httpx.Response(200, content=b"notes")
        assert request.headers["Authorization"] == "Bearer runtime-key"
        if request.url.path.endswith("/download"):
            return httpx.Response(200, json={"url": "https://storage.test/notes.txt"})
        return httpx.Response(200, json={"filename": "notes.txt"})

    monkeypatch.setattr(
        env_module.httpx,
        "AsyncClient",
        partial(
            httpx.AsyncClient,
            transport=httpx.MockTransport(handler),
        ),
    )
    task = env_module.review_files.func(
        prompt="Read the file.",
        attachments=[{"file_id": "notes"}],
        criteria=[],
        hud_api_key="task-key",
    )
    try:
        frame = await anext(task)
        assert frame["data_files"] == [{"path": "files/notes.txt", "file_id": "notes"}]
        assert (tmp_path / "files/notes.txt").read_text() == "notes"
        assert env_module.settings.api_key == "runtime-key"
    finally:
        await task.aclose()
