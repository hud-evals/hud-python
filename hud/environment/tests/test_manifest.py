"""Manifest entries: the task contract published over ``tasks.list``.

``args`` describes the task function's parameters (what stored task args
must satisfy — validated platform-side at sync time); ``input``/``returns``
are the agent's declared I/O types.
"""

from __future__ import annotations

from pydantic import BaseModel

from hud.environment import Environment


class _Point(BaseModel):
    x: int
    y: int


def test_args_schema_captures_params_defaults_and_required() -> None:
    env = Environment("manifests")

    @env.template()
    async def fix_bug(difficulty: int, suite: str = "coding"):
        yield "go"
        yield 1.0

    entry = env.tasks["fix_bug"].manifest_entry()

    schema = entry["args"]
    assert set(schema["properties"]) == {"difficulty", "suite"}
    assert schema["properties"]["difficulty"]["type"] == "integer"
    assert schema["properties"]["suite"]["default"] == "coding"
    assert schema["required"] == ["difficulty"]
    assert schema["additionalProperties"] is False


def test_args_schema_for_no_param_task_rejects_args() -> None:
    env = Environment("manifests")

    @env.template()
    async def bare():
        yield "go"
        yield 1.0

    schema = env.tasks["bare"].manifest_entry()["args"]
    assert schema["properties"] == {}
    assert schema["additionalProperties"] is False


def test_args_schema_var_keyword_allows_additional() -> None:
    env = Environment("manifests")

    @env.template()
    async def flexible(n: int, **rest: str):
        yield "go"
        yield 1.0

    schema = env.tasks["flexible"].manifest_entry()["args"]
    assert set(schema["properties"]) == {"n"}
    assert schema["additionalProperties"] is True


def test_args_schema_unannotated_param_accepts_anything() -> None:
    env = Environment("manifests")

    @env.template()
    async def loose(anything):  # noqa: ANN001
        yield "go"
        yield 1.0

    schema = env.tasks["loose"].manifest_entry()["args"]
    assert schema["required"] == ["anything"]
    assert "type" not in schema["properties"]["anything"]


def test_input_and_returns_schemas_still_published() -> None:
    env = Environment("manifests")

    @env.template(input=_Point, returns=_Point)
    async def typed():
        yield "go"
        yield 1.0

    entry = env.tasks["typed"].manifest_entry()
    assert entry["input"]["properties"]["x"]["type"] == "integer"
    assert entry["returns"]["properties"]["y"]["type"] == "integer"
    assert entry["args"]["properties"] == {}
