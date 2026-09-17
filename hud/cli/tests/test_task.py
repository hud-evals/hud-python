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


@pytest.mark.parametrize("command", ["task", "eval"])
@pytest.mark.parametrize("expose_env", [False, True])
@pytest.mark.parametrize("directory", [False, True])
@pytest.mark.parametrize("outside_source", [False, True])
def test_multifile_source_replays_all_registrations(
    tmp_path, monkeypatch, command, expose_env, directory, outside_source
):
    import sys

    from hud.clients import connect
    from hud.eval import Job
    from hud.settings import settings

    monkeypatch.setattr(settings, "api_key", "test-key")
    monkeypatch.setenv("HUD_TELEMETRY_ENABLED", "false")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "rows.json").write_text('["first", "second"]')
    source_dir = tmp_path / "project" if outside_source else tmp_path
    source_dir.mkdir(exist_ok=True)
    package = source_dir / "split_example"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "core.py").write_text(
        'from pathlib import Path\nfrom hud import Environment\nenv = Environment("split")\n'
        "@env.initialize\nasync def initialize():\n"
        '    assert Path("fixture.txt").read_text() == "answer"\n'
    )
    (package / "fixture.txt").write_text("answer")
    for name in ("first", "second"):
        (package / f"{name}.py").write_text(
            "from pathlib import Path\n"
            "from .core import env\n"
            f'@env.template(id="{name}")\nasync def {name}():\n'
            '    answer = yield "question"\n'
            '    yield 1.0 if answer == Path("fixture.txt").read_text() '
            "and len(env.tasks) == 2 else 0.0\n"
        )
    source = source_dir / "tasks.py"
    source.write_text(
        "import json\nfrom pathlib import Path\n"
        "from split_example.first import first\n"
        "from split_example.second import second\n"
        + ("from split_example.core import env\n" if expose_env else "")
        + 'factories = {"first": first, "second": second}\n'
        + 'tasks = [factories[name]() for name in json.loads(Path("rows.json").read_text())]\n'
    )
    if directory:
        (source_dir / "env.py").write_text("from split_example.core import env\n")
        source = source_dir
    scores = []

    async def run(self, agent, *, runtime, **kwargs):
        for task in self:
            async with runtime(task) as placed, connect(placed) as client:
                await client.start_task(task.id, task.args)
                scores.append((await client.grade({"answer": "answer"}))["score"])
        return Job(id="test-job", name="split")

    if command == "eval":
        monkeypatch.setattr(Taskset, "run", run)
        args = ["eval", str(source), "openai", "--all", "--yes"]
    else:
        args = ["task", "grade", "second", "--source", str(source), "--answer", "answer"]
    try:
        result = CliRunner().invoke(app, [*args, "--json"])
    finally:
        for name in list(sys.modules):
            if name == "split_example" or name.startswith("split_example."):
                sys.modules.pop(name)
    assert result.exit_code == 0, result.output
    if command == "eval":
        assert scores == [1.0, 1.0]
    else:
        assert json.loads(result.stdout)["score"] == 1.0


def test_eval_runs_verifier_imported_only_through_task_source(tmp_path, monkeypatch):
    import sys

    from hud.agents.openai import OpenAIAgent
    from hud.settings import settings

    monkeypatch.setattr(settings, "openai_api_key", "test-key")
    monkeypatch.setenv("HUD_TELEMETRY_ENABLED", "false")
    monkeypatch.chdir(tmp_path)
    package = tmp_path / "verifier_example"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "fixture.txt").write_text("answer")
    (package / "actor.py").write_text(
        'from hud import Environment\nenv = Environment("actor")\n'
        "@env.template()\nasync def solve():\n"
        '    yield "question"\n    yield 0.0\n'
    )
    (package / "judge.py").write_text(
        'from pathlib import Path\nfrom hud import Environment\nenv = Environment("judge")\n'
        "@env.initialize\nasync def initialize():\n"
        '    assert Path("fixture.txt").read_text() == "answer"\n'
        "@env.template()\nasync def verify():\n"
        '    result = yield "verify"\n'
        '    yield 1.0 if result["score"] == 0.0 else 0.0\n'
    )
    source = tmp_path / "tasks.py"
    source.write_text(
        "from verifier_example.actor import solve\n"
        "from verifier_example.judge import verify\n"
        "tasks = [solve()]\ntasks[0].verifier = verify()\n"
    )

    async def answer(self, run):
        run.trace.content = "answer"

    monkeypatch.setattr(OpenAIAgent, "__call__", answer)
    try:
        result = CliRunner().invoke(app, ["eval", str(source), "openai", "--yes", "--json"])
    finally:
        for name in list(sys.modules):
            if name == "verifier_example" or name.startswith("verifier_example."):
                sys.modules.pop(name)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["mean_reward"] == 1.0


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


@pytest.mark.parametrize("suffix", ["json", "py"])
async def test_unbound_source_spawns_the_env_beside_it(tmp_path, monkeypatch, suffix):
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
    if suffix == "py":
        source = tmp_path / "tasks.py"
        source.write_text('from hud.eval import Task\ntasks = [Task(env="example", id="solve")]\n')

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
