"""``Task`` construction, default slug, and serialization round-trips.

``to_dict``/``from_dict`` are the portable identity used by ``hud sync`` and the
JSON/JSONL taskset path, so the tagged env-ref round-trip is the contract under test.
"""

from __future__ import annotations

import json

import pytest

from hud.environment import Environment
from hud.eval import Channel, HudSandbox, RemoteSandbox, Task, Taskset, task
from hud.eval.sandbox import LocalSandbox


def test_task_helper_collects_args_and_metadata() -> None:
    env = Environment("e")
    v = task(env, "task", slug="my-slug", validation=[{"name": "submit"}], x=1, y=2)
    assert v.id == "task"
    assert v.args == {"x": 1, "y": 2}
    assert v.slug == "my-slug"
    assert v.validation == [{"name": "submit"}]


def test_env_task_call_returns_public_task() -> None:
    env = Environment("e")

    @env.task()
    async def solve(n: int):
        yield f"solve:{n}"
        yield 1.0

    runnable = solve(n=3)
    assert isinstance(runnable, Task)
    assert runnable.id == "solve"
    assert runnable.args == {"n": 3}


def test_default_slug_is_task_id_without_args() -> None:
    v = Task(env=Environment("e"), id="solve")
    assert v.default_slug() == "solve"


def test_default_slug_is_deterministic_with_args() -> None:
    env = Environment("e")
    a = Task(env=env, id="solve", args={"b": 2, "a": 1})
    b = Task(env=env, id="solve", args={"a": 1, "b": 2})  # key order differs
    assert a.default_slug() == b.default_slug()  # stable: keys sorted
    assert a.default_slug().startswith("solve-")
    assert a.default_slug() != Task(env=env, id="solve", args={"a": 9}).default_slug()


def test_environment_serializes_to_hud_ref() -> None:
    v = task(Environment("team-intel"), "ask", x=1)
    data = v.to_dict()
    assert data["env"] == {"type": "hud", "name": "team-intel"}
    assert data["task"] == "ask"
    assert data["args"] == {"x": 1}


def test_local_sandbox_unwraps_to_underlying_env_ref() -> None:
    sandbox = LocalSandbox(Environment("wrapped"))
    data = Task(env=sandbox, id="t").to_dict()
    assert data["env"] == {"type": "hud", "name": "wrapped"}


def test_remote_sandbox_serializes_to_url_ref() -> None:
    v = Task(env=RemoteSandbox("tcp://host:7000", token="abc"), id="t")
    data = v.to_dict()
    assert data["env"] == {"type": "url", "url": "tcp://host:7000", "params": {"token": "abc"}}


def test_to_dict_only_includes_set_metadata() -> None:
    data = Task(env=Environment("e"), id="t").to_dict()
    assert set(data) == {"env", "task", "args"}  # no None slug/validation/etc.

    data2 = task(Environment("e"), "t", slug="s", columns={"tier": "easy"}).to_dict()
    assert data2["slug"] == "s"
    assert data2["columns"] == {"tier": "easy"}


def test_roundtrip_is_stable_through_from_dict() -> None:
    original = task(
        Environment("team-intel"),
        "ask",
        slug="ask-v1",
        validation=[{"name": "submit", "arguments": {"answer": "x"}}],
        agent_config={"system_prompt": "be precise"},
        columns={"tier": "hard"},
        difficulty=3,
    ).to_dict()

    rebuilt = Task.from_dict(original)

    assert isinstance(rebuilt.env, HudSandbox)  # hud ref -> HudSandbox
    assert rebuilt.id == "ask"
    assert rebuilt.args == {"difficulty": 3}
    assert rebuilt.slug == "ask-v1"
    assert rebuilt.validation == original["validation"]
    assert rebuilt.agent_config == {"system_prompt": "be precise"}
    assert rebuilt.columns == {"tier": "hard"}
    # ...and re-serializing yields the same portable dict.
    assert rebuilt.to_dict() == original


def test_to_dict_rejects_unserializable_env() -> None:
    class NotAnEnv: ...

    with pytest.raises(TypeError, match="cannot serialize"):
        Task(env=NotAnEnv(), id="t").to_dict()  # type: ignore[arg-type]


def test_from_dict_validates_shape() -> None:
    with pytest.raises(ValueError, match="env"):
        Task.from_dict({"task": "t"})
    with pytest.raises(ValueError, match="task"):
        Task.from_dict({"env": {"type": "hud", "name": "e"}})
    with pytest.raises(ValueError, match="args"):
        Task.from_dict({"env": {"type": "hud", "name": "e"}, "task": "t", "args": "nope"})


def test_taskset_from_tasks_is_ordered_and_keyed_by_slug() -> None:
    env = Environment("e")
    first = task(env, "solve", slug="first", n=1)
    second = task(env, "solve", slug="second", n=2)

    tasks = Taskset.from_tasks("demo", [first, second])

    assert list(tasks) == [first, second]
    assert tasks["first"] is first
    assert list(tasks.filter(["second"])) == [second]
    assert list(tasks.exclude(["first"])) == [second]
    assert list(tasks.items()) == [("first", first), ("second", second)]
    assert tasks.environment_names() == {"e"}


def test_taskset_from_file_loads_json_and_jsonl(tmp_path) -> None:
    env = Environment("e")
    entries = [
        task(env, "solve", slug="one", n=1).to_dict(),
        task(env, "solve", slug="two", n=2).to_dict(),
    ]

    json_path = tmp_path / "tasks.json"
    json_path.write_text(json.dumps(entries), encoding="utf-8")
    jsonl_path = tmp_path / "tasks.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(entry) for entry in entries), encoding="utf-8")

    assert [t.slug for t in Taskset.from_file(json_path)] == ["one", "two"]
    assert [t.slug for t in Taskset.from_file(jsonl_path)] == ["one", "two"]


def test_taskset_to_file_writes_json_jsonl_and_csv(tmp_path) -> None:
    env = Environment("e")
    taskset = Taskset.from_tasks(
        "demo",
        [
            task(env, "solve", slug="one", columns={"tier": "easy"}, n=1),
            task(env, "solve", slug="two", columns={"tier": "hard"}, n={"x": 2}),
        ],
    )

    json_path = taskset.to_file(tmp_path / "tasks.json")
    jsonl_path = taskset.to_file(tmp_path / "tasks.jsonl")
    csv_path = taskset.to_file(tmp_path / "tasks.csv")

    assert [entry["slug"] for entry in json.loads(json_path.read_text())] == ["one", "two"]
    assert [json.loads(line)["slug"] for line in jsonl_path.read_text().splitlines()] == [
        "one",
        "two",
    ]
    csv_text = csv_path.read_text()
    assert "slug,task,env,arg:n,col:tier" in csv_text
    assert "one,solve,e,1,easy" in csv_text
    assert 'two,solve,e,"{""x"": 2}",hard' in csv_text


def test_taskset_from_module_and_package_collect_public_tasks(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = tmp_path / "local_tasks.py"
    module.write_text(
        """
from hud import Environment, task

env = Environment("module-env")
local = task(env, "solve", slug="local", n=1)
""".strip(),
        encoding="utf-8",
    )

    package = tmp_path / "cases"
    case = package / "alpha"
    case.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (case / "__init__.py").write_text("from .task import example\n", encoding="utf-8")
    (case / "task.py").write_text(
        """
from hud import Environment, task

env = Environment("package-env")
example = task(env, "solve", slug="alpha", n=2)
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    assert Taskset.from_module(module)["local"].args == {"n": 1}
    assert Taskset.from_package("cases")["alpha"].args == {"n": 2}


def test_load_environment_selects_by_attr_or_env_name(tmp_path) -> None:
    from hud.eval import load_environment

    module = tmp_path / "envs.py"
    module.write_text(
        """
from hud import Environment

first = Environment("env-one")
second = Environment("env-two")
""".strip(),
        encoding="utf-8",
    )

    assert load_environment(module, name="first").name == "env-one"
    assert load_environment(module, name="env-two").name == "env-two"
    with pytest.raises(ValueError, match="multiple Environments"):
        load_environment(module)
    with pytest.raises(ValueError, match="no Environment named 'missing'"):
        load_environment(module, name="missing")

    single = tmp_path / "single.py"
    single.write_text("from hud import Environment\nenv = Environment('only')\n", encoding="utf-8")
    assert load_environment(single).name == "only"


def test_taskset_from_api_uses_remote_records(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_request(method: str, url: str, **kwargs: object) -> dict[str, object]:
        assert method == "GET"
        if url.endswith("/tasks/evalset/demo"):
            return {"evalset_id": "ts_123", "evalset_name": "Demo"}
        if url.endswith("/tasks/evalsets/ts_123/tasks-by-id"):
            return {
                "evalset_name": "Demo",
                "tasks": {
                    "1": {
                        "env": {"name": "e"},
                        "scenario": "e:solve",
                        "args": {"n": 1},
                        "slug": "one",
                        "column_values": {"tier": "easy"},
                    }
                },
            }
        raise AssertionError(url)

    monkeypatch.setattr("hud.shared.platform.make_request_sync", fake_request)
    monkeypatch.setattr("hud.settings.settings.api_key", "test-key")

    taskset = Taskset.from_api("demo")

    assert taskset.name == "Demo"
    assert taskset["one"].id == "e:solve"
    assert taskset["one"].args == {"n": 1}
    assert taskset["one"].columns == {"tier": "easy"}


def test_taskset_diff_classifies_create_update_unchanged_and_remote_only() -> None:
    env = Environment("e")
    local_a = task(env, "solve", slug="a", n=1)
    local_b = task(env, "solve", slug="b", n=2)
    local_c = task(env, "solve", slug="c", n=3)
    remote_a = Task.from_dict(local_a.to_dict())
    remote_b = task(env, "solve", slug="b", n=99)
    remote_old = task(env, "solve", slug="old", n=0)

    plan = Taskset.from_tasks("demo", [local_a, local_b, local_c]).diff(
        Taskset.from_tasks("demo", [remote_a, remote_b, remote_old]),
    )

    assert [t.slug for t in plan.to_create] == ["c"]
    assert [t.slug for t in plan.to_update] == ["b"]
    assert [t.slug for t in plan.unchanged] == ["a"]
    assert [t.slug for t in plan.remote_only] == ["old"]
    assert "Create: 1" in plan.summary()


def test_upload_taskset_posts_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    from hud.eval.taskset import taskset_column_definitions, upload_taskset
    from hud.shared.platform import PlatformClient

    env = Environment("e")
    upload = task(env, "solve", slug="solve-one", columns={"tier": "easy"}, n=1)
    posted: dict[str, object] = {}

    def fake_request(method: str, url: str, json: object = None, **kwargs: object) -> dict:
        posted.update(method=method, url=url, json=json, api_key=kwargs.get("api_key"))
        return {"ok": True}

    monkeypatch.setattr("hud.shared.platform.make_request_sync", fake_request)

    platform = PlatformClient("https://api.example", "token")
    result = upload_taskset(
        platform, "demo", [upload], columns=taskset_column_definitions([upload])
    )

    assert result == {"ok": True}
    assert posted["method"] == "POST"
    assert posted["url"] == "https://api.example/tasks/upload"
    assert posted["api_key"] == "token"
    assert posted["json"] == {
        "name": "demo",
        "tasks": [
            {
                "slug": "solve-one",
                "env": {"name": "e"},
                "scenario": "e:solve",
                "args": {"n": 1},
                "column_values": {"tier": "easy"},
            },
        ],
        "columns": {"tier": {"type": "text"}},
    }


async def test_remote_sandbox_create_returns_channel() -> None:
    sandbox = RemoteSandbox("tcp://host:7000", token="abc")

    channel = await sandbox.create()

    assert isinstance(channel, Channel)
    assert channel.url == "tcp://host:7000"
    assert channel.params == {"token": "abc"}
    assert sandbox.channel is channel

    await sandbox.terminate()
    with pytest.raises(RuntimeError, match="not created"):
        _ = sandbox.channel
