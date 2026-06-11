"""``ensure_strict_json_schema`` — coerce a JSON schema to OpenAI strict-mode form."""

from __future__ import annotations

from typing import Any

from hud.agents.openai.tools.strict_schema import ensure_strict_json_schema


def test_empty_schema_becomes_closed_object() -> None:
    result = ensure_strict_json_schema({})
    assert result == {
        "additionalProperties": False,
        "type": "object",
        "properties": {},
        "required": [],
    }


def test_object_gets_additional_properties_false_and_all_required() -> None:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
    }
    result = ensure_strict_json_schema(schema)

    assert result["additionalProperties"] is False
    assert set(result["required"]) == {"a", "b"}  # strict mode requires every property


def test_additional_properties_true_is_converted_to_false() -> None:
    result = ensure_strict_json_schema(
        {"type": "object", "properties": {}, "additionalProperties": True}
    )
    assert result["additionalProperties"] is False


def test_unsupported_keywords_are_stripped() -> None:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "title": "Name",  # unsupported meta keyword
                "minLength": 1,  # unsupported string constraint
            },
        },
    }
    name_schema = ensure_strict_json_schema(schema)["properties"]["name"]
    assert "title" not in name_schema
    assert "minLength" not in name_schema
    assert name_schema["type"] == "string"


def test_nested_objects_are_recursively_strict() -> None:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "inner": {"type": "object", "properties": {"x": {"type": "number"}}},
        },
    }
    inner = ensure_strict_json_schema(schema)["properties"]["inner"]
    assert inner["additionalProperties"] is False
    assert inner["required"] == ["x"]


def test_is_idempotent() -> None:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"a": {"type": "string", "title": "A"}},
    }
    once = ensure_strict_json_schema(dict(schema))
    twice = ensure_strict_json_schema(once)
    assert once == twice
