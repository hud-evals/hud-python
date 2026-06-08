"""``Variant`` construction, default slug, and serialization round-trips.

``to_dict``/``from_dict`` are the portable identity used by ``hud sync`` and the
JSON/JSONL taskset path, so the tagged env-ref round-trip is the contract under test.
"""

from __future__ import annotations

import pytest

from hud.environment import Environment
from hud.eval import RemoteSandbox, Variant, variant
from hud.eval.sandbox import LocalSandbox


def test_variant_helper_collects_args_and_metadata() -> None:
    env = Environment("e")
    v = variant(env, "task", slug="my-slug", validation=[{"name": "submit"}], x=1, y=2)
    assert v.task == "task"
    assert v.args == {"x": 1, "y": 2}
    assert v.slug == "my-slug"
    assert v.validation == [{"name": "submit"}]


def test_default_slug_is_task_id_without_args() -> None:
    v = Variant(env=Environment("e"), task="solve")
    assert v.default_slug() == "solve"


def test_default_slug_is_deterministic_with_args() -> None:
    env = Environment("e")
    a = Variant(env=env, task="solve", args={"b": 2, "a": 1})
    b = Variant(env=env, task="solve", args={"a": 1, "b": 2})  # key order differs
    assert a.default_slug() == b.default_slug()  # stable: keys sorted
    assert a.default_slug().startswith("solve-")
    assert a.default_slug() != Variant(env=env, task="solve", args={"a": 9}).default_slug()


def test_environment_serializes_to_hud_ref() -> None:
    v = variant(Environment("team-intel"), "ask", x=1)
    data = v.to_dict()
    assert data["env"] == {"type": "hud", "name": "team-intel"}
    assert data["task"] == "ask"
    assert data["args"] == {"x": 1}


def test_local_sandbox_unwraps_to_underlying_env_ref() -> None:
    sandbox = LocalSandbox(Environment("wrapped"))
    data = Variant(env=sandbox, task="t").to_dict()
    assert data["env"] == {"type": "hud", "name": "wrapped"}


def test_remote_sandbox_serializes_to_url_ref() -> None:
    v = Variant(env=RemoteSandbox("tcp://host:7000", auth_token="abc"), task="t")
    data = v.to_dict()
    assert data["env"] == {"type": "url", "url": "tcp://host:7000"}


def test_to_dict_only_includes_set_metadata() -> None:
    data = Variant(env=Environment("e"), task="t").to_dict()
    assert set(data) == {"env", "task", "args"}  # no None slug/validation/etc.

    data2 = variant(Environment("e"), "t", slug="s", columns={"tier": "easy"}).to_dict()
    assert data2["slug"] == "s"
    assert data2["columns"] == {"tier": "easy"}


def test_roundtrip_is_stable_through_from_dict() -> None:
    original = Variant(
        env=RemoteSandbox("tcp://host:7000", auth_token="secret"),
        task="ask",
        args={"difficulty": 3},
        slug="ask-v1",
        validation=[{"name": "submit", "arguments": {"answer": "x"}}],
        agent_config={"system_prompt": "be precise"},
        columns={"tier": "hard"},
    ).to_dict()

    rebuilt = Variant.from_dict(original)

    assert isinstance(rebuilt.env, RemoteSandbox)
    assert rebuilt.task == "ask"
    assert rebuilt.args == {"difficulty": 3}
    assert rebuilt.slug == "ask-v1"
    assert rebuilt.validation == original["validation"]
    assert rebuilt.agent_config == {"system_prompt": "be precise"}
    assert rebuilt.columns == {"tier": "hard"}
    # ...and re-serializing yields the same portable dict.
    assert rebuilt.to_dict() == original


def test_hud_ref_is_registry_identity_not_runnable_sandbox() -> None:
    with pytest.raises(ValueError, match="not runnable locally"):
        Variant.from_dict({"env": {"type": "hud", "name": "e"}, "task": "t"})


def test_to_dict_rejects_unserializable_env() -> None:
    class NotAnEnv: ...

    with pytest.raises(TypeError, match="cannot serialize"):
        Variant(env=NotAnEnv(), task="t").to_dict()  # type: ignore[arg-type]


def test_from_dict_validates_shape() -> None:
    with pytest.raises(ValueError, match="env"):
        Variant.from_dict({"task": "t"})
    with pytest.raises(ValueError, match="task"):
        Variant.from_dict({"env": {"type": "hud", "name": "e"}})
    with pytest.raises(ValueError, match="args"):
        Variant.from_dict({"env": {"type": "hud", "name": "e"}, "task": "t", "args": "nope"})
