"""Schema utilities for tool definitions."""

from __future__ import annotations

from typing import Any

__all__ = ["ensure_strict_schema", "json_type_to_python", "schema_to_pydantic"]


def ensure_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Ensure a JSON schema is compatible with OpenAI's strict mode.

    OpenAI strict mode requires:
    - additionalProperties: false on all objects
    - All properties must be in required

    Args:
        schema: Original JSON schema.

    Returns:
        Modified schema for strict mode.
    """
    schema = dict(schema)

    if schema.get("type") == "object":
        schema["additionalProperties"] = False

        if "properties" in schema:
            # All properties must be required
            schema["required"] = list(schema["properties"].keys())

            # Recursively process nested objects
            for prop_schema in schema["properties"].values():
                if isinstance(prop_schema, dict):
                    _ensure_strict_recursive(prop_schema)

    return schema


def _ensure_strict_recursive(schema: dict[str, Any]) -> None:
    """Recursively apply strict mode to nested schemas."""
    if schema.get("type") == "object":
        schema["additionalProperties"] = False
        if "properties" in schema:
            schema["required"] = list(schema["properties"].keys())
            for prop_schema in schema["properties"].values():
                if isinstance(prop_schema, dict):
                    _ensure_strict_recursive(prop_schema)

    elif schema.get("type") == "array" and "items" in schema:
        if isinstance(schema["items"], dict):
            _ensure_strict_recursive(schema["items"])


def schema_to_pydantic(name: str, schema: dict[str, Any]) -> type:
    """Convert JSON schema to a Pydantic model.

    Args:
        name: Model name (used for class name).
        schema: JSON schema with properties.

    Returns:
        Dynamically created Pydantic model class.
    """
    from pydantic import Field, create_model

    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    fields = {}
    for prop_name, prop_schema in properties.items():
        prop_type = json_type_to_python(prop_schema.get("type", "string"))
        default = ... if prop_name in required else None
        description = prop_schema.get("description", "")
        fields[prop_name] = (prop_type, Field(default=default, description=description))

    return create_model(f"{name}Input", **fields)


def json_type_to_python(json_type: str) -> type:
    """Map JSON schema type to Python type.

    Args:
        json_type: JSON schema type string.

    Returns:
        Corresponding Python type.
    """
    mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    return mapping.get(json_type, str)
