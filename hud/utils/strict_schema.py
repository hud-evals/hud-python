"""Utilities to convert JSON schemas into OpenAI's strict format."""

from __future__ import annotations

from typing import Any, TypeGuard

_EMPTY_SCHEMA = {
    "additionalProperties": False,
    "type": "object",
    "properties": {},
    "required": [],
}


def ensure_strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Ensure a JSON schema conforms to OpenAI's strict requirements.

    This mutates the provided schema in-place and returns it for convenience.
    """
    if schema == {}:
        return _EMPTY_SCHEMA.copy()
    return _ensure_strict_json_schema(schema, path=(), root=schema)


def _ensure_strict_json_schema(
    json_schema: object,
    *,
    path: tuple[str, ...],
    root: dict[str, Any],
) -> dict[str, Any]:
    if not _is_dict(json_schema):
        raise TypeError(f"Expected {json_schema} to be a dictionary; path={path}")

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

    typ = json_schema.get("type")
    if typ == "object":
        if "additionalProperties" not in json_schema or json_schema["additionalProperties"] is True:
            json_schema["additionalProperties"] = False
        elif (
            json_schema["additionalProperties"] and json_schema["additionalProperties"] is not False
        ):
            raise ValueError(
                "additionalProperties should not be set for object types in strict mode."
            )

    properties = json_schema.get("properties")
    if _is_dict(properties):
        json_schema["required"] = list(properties.keys())
        json_schema["properties"] = {
            key: _ensure_strict_json_schema(prop_schema, path=(*path, "properties", key), root=root)
            for key, prop_schema in properties.items()
        }

    items = json_schema.get("items")
    if _is_dict(items):
        json_schema["items"] = _ensure_strict_json_schema(items, path=(*path, "items"), root=root)

    prefix_items = json_schema.get("prefixItems")
    if _is_list(prefix_items) and prefix_items:
        item_types = set()
        for item in prefix_items:
            if _is_dict(item) and "type" in item:
                item_types.add(item["type"])

        if len(item_types) == 1:
            item_type = item_types.pop()
            json_schema["items"] = {"type": item_type}
        else:
            json_schema["items"] = {"type": "integer"}

        tuple_length = len(prefix_items)
        json_schema["minItems"] = tuple_length
        json_schema["maxItems"] = tuple_length
        json_schema.pop("prefixItems")

    any_of = json_schema.get("anyOf")
    if _is_list(any_of):
        json_schema["anyOf"] = [
            _ensure_strict_json_schema(variant, path=(*path, "anyOf", str(i)), root=root)
            for i, variant in enumerate(any_of)
        ]

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

    if "default" in json_schema:
        json_schema.pop("default")

    for keyword in ("title", "examples"):
        json_schema.pop(keyword, None)

    ref = json_schema.get("$ref")
    if ref and _has_more_than_n_keys(json_schema, 1):
        if not isinstance(ref, str):
            raise ValueError(f"Received non-string $ref - {ref}")
        resolved = _resolve_ref(root=root, ref=ref)
        if not _is_dict(resolved):
            raise ValueError(
                f"Expected `$ref: {ref}` to resolve to a dictionary but got {resolved}"
            )
        json_schema.update({**resolved, **json_schema})
        json_schema.pop("$ref")
        return _ensure_strict_json_schema(json_schema, path=path, root=root)

    return json_schema


def _resolve_ref(*, root: dict[str, Any], ref: str) -> object:
    if not ref.startswith("#/"):
        raise ValueError(f"Unexpected $ref format {ref!r}; does not start with #/")

    path = ref[2:].split("/")
    resolved: object = root
    for key in path:
        assert _is_dict(resolved), f"Encountered non-dictionary entry while resolving {ref}"
        resolved = resolved[key]

    return resolved


def _is_dict(obj: object) -> TypeGuard[dict[str, Any]]:
    return isinstance(obj, dict)


def _is_list(obj: object) -> TypeGuard[list[object]]:
    return isinstance(obj, list)


def _has_more_than_n_keys(obj: dict[str, object], n: int) -> bool:
    return any(count > n for count, _ in enumerate(obj, start=1))
