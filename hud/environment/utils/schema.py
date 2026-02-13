"""Schema utilities for tool definitions."""

from __future__ import annotations

from typing import Any

__all__ = [
    "json_type_to_python",
    "schema_to_pydantic",
]


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
