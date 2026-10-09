import asyncio
import json

import pytest
from typer.testing import CliRunner

from hud.cli import task as task_module
from hud.cli.__main__ import app
from hud.eval import Task, Taskset


@pytest.mark.parametrize("override", [None, {}])
async def test_source_resolves_authored_task_for_existing_runtime(tmp_path, override):
    authored = Task(
        env="coding",
        id="coding-task",
        slug="flask-4992",
        args={"description": "Fix Flask", "test_script": "pytest"},
    )
    source = Taskset("authored", [authored]).to_file(tmp_path / "tasks.json")

    task_id, args, placement = task_module._resolve(
        "flask-4992",
        str(source),
        "tcp://127.0.0.1:9000",
        override,
    )

    assert task_id == "coding-task"
    assert args == (authored.args if override is None else override)
    async with placement as runtime:
        assert runtime.url == "tcp://127.0.0.1:9000"


async def test_url_without_source_uses_raw_task_and_args(monkeypatch):
    def fail(cls, source):
        raise AssertionError(f"unexpected task source: {source}")

    monkeypatch.setattr(Taskset, "from_file", classmethod(fail))

    task_id, args, placement = task_module._resolve(
        "coding-task",
        None,
        "tcp://127.0.0.1:9000",
        {"description": "Fix Flask"},
    )

    assert task_id == "coding-task"
    assert args == {"description": "Fix Flask"}
    async with placement as runtime:
        assert runtime.url == "tcp://127.0.0.1:9000"


async def test_task_source_uses_sibling_environment_for_start_and_grade(tmp_path, monkeypatch):
    import sys

    from hud.clients import connect

    monkeypatch.setenv("HUD_TELEMETRY_ENABLED", "false")
    monkeypatch.delitem(sys.modules, "env", raising=False)
    (tmp_path / "env.py").write_text(
        'from hud import Environment\nenv = Environment("example")\n'
        '@env.template(id="solve")\nasync def solve():\n'
        '    answer = yield "question"\n    yield 1.0 if answer == "answer" else 0.0\n'
    )
    source = tmp_path / "tasks.py"
    source.write_text("from env import solve\n\ntasks = [solve()]\n")
    try:
        task_id, args, placement = task_module._resolve("solve", str(source), None, {})
        async with placement as runtime, connect(runtime) as client:
            await client.start_task(task_id, args)
            result = await client.grade({"answer": "answer"})
    finally:
        sys.modules.pop("env", None)
    assert result["score"] == 1.0


@pytest.mark.parametrize("mode", ["empty", "parked", "failed_grade", "ambiguous"])
async def test_grade_only_starts_when_no_task_is_in_progress(mode):
    from hud.clients import connect
    from hud.environment import Environment
    from hud.eval import LocalRuntime

    env = Environment("grading")
    starts = 0

    @env.template()
    async def solve():
        nonlocal starts
        starts += 1
        yield "question"
        if mode == "failed_grade":
            raise ValueError("grader failed")
        yield 1.0

    async with LocalRuntime(env)(Task(env="grading", id="solve")) as runtime:
        for _ in range(2 if mode == "ambiguous" else int(mode != "empty")):
            async with connect(runtime) as client:
                await client.start_task("solve", {})
        result = await asyncio.to_thread(
            CliRunner().invoke,
            app,
            ["task", "grade", "solve", "--url", runtime.url, "--json"],
        )
    assert starts == (2 if mode == "ambiguous" else 1)
    if mode in {"empty", "parked"}:
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["score"] == 1.0
    else:
        assert result.exit_code != 0
        assert (
            "grader failed" in result.output
            if mode == "failed_grade"
            else "2 parked sessions" in result.output
        )


async def test_json_source_spawns_the_env_beside_it(tmp_path, monkeypatch):
    from hud.clients import connect

    monkeypatch.setenv("HUD_TELEMETRY_ENABLED", "false")
    (tmp_path / "env.py").write_text(
        'from hud import Environment\nenv = Environment("example")\n'
        '@env.template(id="solve")\nasync def solve():\n'
        '    answer = yield "question"\n    yield 1.0 if answer == "answer" else 0.0\n'
    )
    source = Taskset(
        "authored",
        [Task(env="example", id="solve", slug="solve")],
    ).to_file(tmp_path / "tasks.json")

    task_id, args, placement = task_module._resolve("solve", str(source), None, None)
    async with placement as runtime, connect(runtime) as client:
        await client.start_task(task_id, args)
        result = await client.grade({"answer": "answer"})
    assert result["score"] == 1.0


def test_task_id_matching_multiple_rows_requires_unique_slug(tmp_path):
    source = Taskset(
        "authored",
        [
            Task(env="example", id="solve", slug="first"),
            Task(env="example", id="solve", slug="second"),
        ],
    ).to_file(tmp_path / "tasks.json")
    result = CliRunner().invoke(app, ["task", "start", "solve", "--source", str(source), "--json"])
    assert result.exit_code == 2
    assert "Ambiguous task" in json.loads(result.stdout)["message"]
