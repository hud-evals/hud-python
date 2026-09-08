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
