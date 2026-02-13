"""Ensure JSON schemas conform to OpenAI's strict structured outputs format.

Adapted from https://github.com/openai/openai-agents-python/blob/main/src/agents/strict_schema.py
with additional handling for MCP tool schemas (prefixItems, additionalProperties: true,
and comprehensive unsupported keyword stripping).
"""

from __future__ import annotations

from typing import Any, TypeGuard

_EMPTY_SCHEMA: dict[str, Any] = {
    "additionalProperties": False,
    "type": "object",
    "properties": {},
    "required": [],
}

# Keywords not supported by OpenAI's strict structured outputs.
# See: https://platform.openai.com/docs/guides/structured-outputs/supported-schemas
_UNSUPPORTED_KEYWORDS = frozenset(
    [
        # Meta keywords
        "$schema",
        "$id",
        "$comment",
        "title",
        "examples",
        "deprecated",
        # String constraints
        "minLength",
        "maxLength",
        "pattern",
        "format",
        # Number constraints
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        # Object constraints
        "patternProperties",
        "unevaluatedProperties",
        "propertyNames",
        "minProperties",
        "maxProperties",
        # Array constraints
        "minItems",
        "maxItems",
        "uniqueItems",
        "unevaluatedItems",
        "contains",
        "minContains",
        "maxContains",
        # Conditional
        "if",
        "then",
        "else",
        "dependentRequired",
        "dependentSchemas",
        "not",
        # Access modifiers
        "readOnly",
        "writeOnly",
        # Content
        "contentMediaType",
        "contentEncoding",
    ]
)


def ensure_strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Mutate a JSON schema so it conforms to OpenAI's strict structured outputs.

    Returns the (mutated) schema for convenience.
    """
    if schema == {}:
        return _EMPTY_SCHEMA.copy()
    return _ensure_strict_json_schema(schema, path=(), root=schema)


def _ensure_strict_json_schema(
    json_schema: object,
    *,
    path: tuple[str, ...],
    root: dict[str, object],
) -> dict[str, Any]:
    if not _is_dict(json_schema):
        raise TypeError(f"Expected {json_schema} to be a dictionary; path={path}")

    # --- $defs / definitions ---
    defs = json_schema.get("$defs")
    if _is_dict(defs):
        for def_name, def_schema in defs.items():
            _ensure_strict_json_schema(def_schema, path=(*path, "$defs", def_name), root=root)

    definitions = json_schema.get("definitions")
    if _is_dict(definitions):
        for definition_name, definition_schema in definitions.items():
            _ensure_strict_json_schema(
                definition_schema, path=(*path, "definitions", definition_name), root=root
            )

    # --- object: additionalProperties ---
    typ = json_schema.get("type")
    if typ == "object":
        if "additionalProperties" not in json_schema:
            json_schema["additionalProperties"] = False
        elif json_schema["additionalProperties"] is True:
            # MCP schemas commonly allow extra fields; silently convert for strict mode.
            json_schema["additionalProperties"] = False
        elif (
            json_schema["additionalProperties"] and json_schema["additionalProperties"] is not False
        ):
            # additionalProperties is a schema object — not allowed in strict mode.
            raise ValueError(
                "additionalProperties should not be set for object types. This could be because "
                "you're using an older version of Pydantic, or because you configured additional "
                "properties to be allowed. If you really need this, update the function or output "
                "tool to not use a strict schema."
            )

    # --- object: properties → required ---
    properties = json_schema.get("properties")
    if _is_dict(properties):
        json_schema["required"] = list(properties.keys())
        json_schema["properties"] = {
            key: _ensure_strict_json_schema(prop_schema, path=(*path, "properties", key), root=root)
            for key, prop_schema in properties.items()
        }

    # --- array: items ---
    items = json_schema.get("items")
    if _is_dict(items):
        json_schema["items"] = _ensure_strict_json_schema(items, path=(*path, "items"), root=root)

    # --- prefixItems (tuple schemas) → plain array ---
    # prefixItems, minItems, maxItems are NOT supported in strict mode.
    prefix_items = json_schema.get("prefixItems")
    if _is_list(prefix_items) and prefix_items:
        item_types = set()
        for item in prefix_items:
            if _is_dict(item) and "type" in item:
                item_types.add(item["type"])

        if len(item_types) == 1:
            json_schema["items"] = {"type": item_types.pop()}
        else:
            # Mixed types or complex schemas — use integer as fallback (common for coordinates).
            json_schema["items"] = {"type": "integer"}

        json_schema.pop("prefixItems")

    # --- anyOf ---
    any_of = json_schema.get("anyOf")
    if _is_list(any_of):
        json_schema["anyOf"] = [
            _ensure_strict_json_schema(variant, path=(*path, "anyOf", str(i)), root=root)
            for i, variant in enumerate(any_of)
        ]

    # --- oneOf → anyOf (oneOf unsupported in nested contexts) ---
    one_of = json_schema.get("oneOf")
    if _is_list(one_of):
        existing_any_of = json_schema.get("anyOf", [])
        if not _is_list(existing_any_of):
            existing_any_of = []
        json_schema["anyOf"] = existing_any_of + [
            _ensure_strict_json_schema(variant, path=(*path, "oneOf", str(i)), root=root)
            for i, variant in enumerate(one_of)
        ]
        json_schema.pop("oneOf")

    # --- allOf ---
    all_of = json_schema.get("allOf")
    if _is_list(all_of):
        if len(all_of) == 1:
            json_schema.update(
                _ensure_strict_json_schema(all_of[0], path=(*path, "allOf", "0"), root=root)
            )
            json_schema.pop("allOf")
        else:
            json_schema["allOf"] = [
                _ensure_strict_json_schema(entry, path=(*path, "allOf", str(i)), root=root)
                for i, entry in enumerate(all_of)
            ]

    # --- Strip defaults (not supported in strict mode) ---
    json_schema.pop("default", None)

    # --- Strip unsupported keywords ---
    for keyword in _UNSUPPORTED_KEYWORDS:
        json_schema.pop(keyword, None)

    # --- Unravel $ref when mixed with other keys ---
    ref = json_schema.get("$ref")
    if ref and _has_more_than_n_keys(json_schema, 1):
        assert isinstance(ref, str), f"Received non-string $ref - {ref}"

        resolved = _resolve_ref(root=root, ref=ref)
        if not _is_dict(resolved):
            raise ValueError(
                f"Expected `$ref: {ref}` to resolved to a dictionary but got {resolved}"
            )

        json_schema.update({**resolved, **json_schema})
        json_schema.pop("$ref")
        return _ensure_strict_json_schema(json_schema, path=path, root=root)

    return json_schema


def _resolve_ref(*, root: dict[str, object], ref: str) -> object:
    if not ref.startswith("#/"):
        raise ValueError(f"Unexpected $ref format {ref!r}; Does not start with #/")

    path = ref[2:].split("/")
    resolved = root
    for key in path:
        value = resolved[key]
        assert _is_dict(value), (
            f"encountered non-dictionary entry while resolving {ref} - {resolved}"
        )
        resolved = value

    return resolved


def _is_dict(obj: object) -> TypeGuard[dict[str, object]]:
    return isinstance(obj, dict)


def _is_list(obj: object) -> TypeGuard[list[object]]:
    return isinstance(obj, list)


def _has_more_than_n_keys(obj: dict[str, object], n: int) -> bool:
    return any(i > n for i, _ in enumerate(obj, start=1))
