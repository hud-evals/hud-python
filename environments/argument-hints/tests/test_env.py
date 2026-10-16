"""What the example promises: hinted arguments, safe staging, weighted grading, the runtime key.

Each test starts a fresh environment process against the stand-in platform in
``conftest.py``, starts a task with the arguments a task row carries, and grades
an answer: what a rollout does, without an agent.
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from hud import SubprocessRuntime, Task, connect
from hud.clients import HudProtocolError

from env import review_files

ENV_SOURCE = Path(__file__).resolve().parent.parent / "env.py"
CRITERIA = [
    {"requirement": "Names exactly three primes.", "weight": 2},
    {"requirement": "Explains each one.", "weight": 1},
    {"requirement": "Calls a composite number prime.", "weight": -2},
]


def row(attachments: list[dict] | None = None, criteria: list[dict] | None = None) -> Task:
    return review_files(
        prompt="Name three primes.",
        attachments=attachments or [],
        criteria=CRITERIA if criteria is None else criteria,
        hud_api_key="task-key",
    )


@asynccontextmanager
async def served(task: Task):
    """A fresh environment process serving ``task``, and a client connected to it."""
    async with SubprocessRuntime(ENV_SOURCE)(task) as runtime, connect(runtime) as client:
        yield client


async def test_every_editable_argument_declares_its_hint():
    async with served(row()) as client:
        (template,) = await client.list_tasks()

    arguments = template["args"]
    hints = {name: spec.get("x-hud-hint") for name, spec in arguments["properties"].items()}
    assert hints == {
        "prompt": "prompt",
        "attachments": "data-files",
        "criteria": "grading",
        "hud_api_key": None,
    }
    assert "hud_api_key" not in arguments["required"]
    # The console resolves the attachment reference through $defs.
    assert arguments["properties"]["attachments"]["items"]["$ref"] == "#/$defs/DataFileRef"
    assert "file_id" in arguments["$defs"]["DataFileRef"]["properties"]


async def test_attachments_are_staged_with_the_runtime_key_and_declared(platform, workspace):
    platform.upload("notes", "notes.txt", b"three primes")
    platform.upload("report", "report.pdf", b"%PDF")
    task = row(attachments=[{"file_id": "notes", "path": "reading/notes.md"}, {"file_id": "report"}])

    async with served(task) as client:
        started = await client.start_task(task.id, task.args)

    assert started["data_files"] == [
        {"path": "files/reading/notes.md", "file_id": "notes"},
        {"path": "files/report.pdf", "file_id": "report"},
    ]
    assert "- files/reading/notes.md\n- files/report.pdf" in started["prompt"]
    assert started["prompt"].endswith("Name three primes.")
    assert (workspace / "files/reading/notes.md").read_bytes() == b"three primes"
    assert (workspace / "files/report.pdf").read_bytes() == b"%PDF"
    assert {item.authorization for item in platform.requests("GET", "/v2/data")} == {"Bearer runtime-key"}
    # The presigned URL carries its own credentials; the key never goes to storage.
    assert {item.authorization for item in platform.requests("GET", "/storage")} == {None}


@pytest.mark.parametrize(
    ("upload", "attachment", "message"),
    [
        pytest.param(
            {"filename": "notes.txt"},
            {"file_id": "notes", "path": "../escape.md"},
            "unsafe data file path",
            id="hostile-destination",
        ),
        pytest.param(
            {"filename": "../../escape.md"},
            {"file_id": "notes"},
            "unsafe data file path",
            id="hostile-uploaded-filename",
        ),
        pytest.param(
            {"filename": "..\\escape.md"},
            {"file_id": "notes"},
            "unsafe data file path",
            id="backslash-filename",
        ),
        pytest.param(None, {"file_id": "notes"}, "GET /v2/data/notes failed: 404", id="unknown-file"),
        pytest.param(
            {"filename": "notes.txt", "downloadable": False},
            {"file_id": "notes"},
            "has no download url",
            id="no-download-url",
        ),
    ],
)
async def test_a_file_that_cannot_be_staged_fails_the_start_before_anything_is_written(
    upload, attachment, message, platform, workspace
):
    if upload is not None:
        platform.upload("notes", **upload)
    task = row(attachments=[attachment])

    async with served(task) as client:
        with pytest.raises(HudProtocolError) as failure:
            await client.start_task(task.id, task.args)

    assert message in failure.value.message
    assert platform.requests("GET", "/storage") == []
    assert list((workspace / "files").rglob("*")) == []
    assert not (workspace / "escape.md").exists()


async def test_staging_without_a_hud_key_fails_the_start(monkeypatch, platform):
    monkeypatch.setenv("HUD_API_KEY", "")
    platform.upload("notes", "notes.txt")
    task = row(attachments=[{"file_id": "notes"}])

    async with served(task) as client:
        with pytest.raises(HudProtocolError) as failure:
            await client.start_task(task.id, task.args)

    assert "HUD_API_KEY is unset" in failure.value.message
    assert platform.received == []


async def test_a_previous_task_leaves_nothing_in_the_files_directory(workspace):
    files = workspace / "files"
    files.mkdir(parents=True)
    (files / "escape").symlink_to(workspace.parent / "outside")
    task = row()

    async with served(task) as client:
        started = await client.start_task(task.id, task.args)

    assert started["data_files"] == []
    assert "- (none)" in started["prompt"]
    assert list(files.iterdir()) == []


@pytest.mark.parametrize(
    ("criteria", "score", "judged"),
    [
        pytest.param(
            CRITERIA,
            1.0,
            [
                ("positive", "Names exactly three primes."),
                ("positive", "Explains each one."),
                ("negative", "Calls a composite number prime."),
            ],
            id="weighted-criteria",
        ),
        pytest.param([], 0.0, [], id="no-criteria"),
    ],
)
async def test_the_criteria_reach_the_judge_unchanged(criteria, score, judged, platform):
    task = row(criteria=criteria)

    async with served(task) as client:
        await client.start_task(task.id, task.args)
        grade = await client.grade({"answer": "2, 3 and 5"})

    assert grade["score"] == pytest.approx(score)
    prompts = [item.body["messages"][-1]["content"] for item in platform.requests("POST", "/")]
    assert sorted(
        (prompt.split("<criterion_type>\n")[1].split("\n")[0], prompt.split("<criterion>\n")[1].split("\n")[0])
        for prompt in prompts
    ) == sorted(judged)
    assert all("<response>\n2, 3 and 5\n</response>" in prompt for prompt in prompts)
    assert {item.authorization for item in platform.requests("POST", "/")} <= {"Bearer runtime-key"}


@pytest.mark.parametrize("rejected", [set(), {"second"}], ids=["both-graded", "one-judge-rejects"])
async def test_concurrent_sessions_grade_with_the_runtime_key(rejected, platform):
    platform.rejected_answers = rejected
    platform.judging = threading.Barrier(2)
    first = row(criteria=[{"requirement": "An answer is present.", "weight": 1}])
    second = first.model_copy(update={"args": {**first.args, "hud_api_key": "second-key"}})

    async with (
        SubprocessRuntime(ENV_SOURCE)(first) as runtime,
        connect(runtime) as one,
        connect(runtime) as two,
    ):
        await one.start_task(first.id, first.args)
        await two.start_task(second.id, second.args)
        grades = await asyncio.gather(
            one.grade({"answer": "first"}), two.grade({"answer": "second"}), return_exceptions=True
        )

    assert {item.authorization for item in platform.requests("POST", "/")} == {"Bearer runtime-key"}
    assert grades[0]["score"] == 1.0
    if rejected:
        assert isinstance(grades[1], HudProtocolError)
        assert "judge rejected request" in grades[1].message
    else:
        assert grades[1]["score"] == 1.0
    assert "task-key" not in str(grades) and "second-key" not in str(grades)
