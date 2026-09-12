from hud.cli import task as task_module
from hud.eval import Task, Taskset


async def test_source_resolves_authored_task_for_existing_runtime(monkeypatch):
    authored = Task(
        env="coding",
        id="coding-task",
        slug="flask-4992",
        args={"description": "Fix Flask", "test_script": "pytest"},
    )
    monkeypatch.setattr(task_module, "_collect", lambda source: Taskset(source, [authored]))

    task_id, args, placement = task_module._resolve(
        "flask-4992",
        "tasks.py",
        "127.0.0.1:9000",
        {},
    )

    assert task_id == "coding-task"
    assert args == authored.args
    async with placement as runtime:
        assert runtime.url == "tcp://127.0.0.1:9000"


async def test_url_without_source_uses_raw_task_and_args(monkeypatch):
    def fail(source):
        raise AssertionError(f"unexpected task source: {source}")

    monkeypatch.setattr(task_module, "_collect", fail)

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
    from hud.clients import connect

    monkeypatch.setenv("HUD_TELEMETRY_ENABLED", "false")
    (tmp_path / "env.py").write_text(
        'from hud import Environment\nenv = Environment("example")\n'
        '@env.template(id="solve")\nasync def solve():\n'
        '    answer = yield "question"\n    yield 1.0 if answer == "answer" else 0.0\n'
    )
    source = tmp_path / "tasks.py"
    source.write_text('from hud.eval import Task\ntasks = [Task(env="example", id="solve")]\n')
    task_id, args, placement = task_module._resolve("solve", str(source), None, {})
    async with placement as runtime, connect(runtime) as client:
        await client.start_task(task_id, args)
        result = await client.grade({"answer": "answer"})
    assert result["score"] == 1.0
