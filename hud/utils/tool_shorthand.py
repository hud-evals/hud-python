from __future__ import annotations

from typing import Any


def _is_call_like(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    if "name" in obj and "arguments" in obj:
        return True
    if len(obj) == 1:
        _, v = next(iter(obj.items()))
        if isinstance(v, dict):
            return "name" in v or (len(v) == 1 and isinstance(next(iter(v.values())), dict))
    return False


def _to_call_dict(obj: Any) -> Any:
    """Recursively convert shorthand/wrapped dicts into name/arguments templates.

    Rules:
    - If obj is a dict with {name, arguments}: return {name, arguments: recurse(arguments)}
    - Else if obj is a single-key dict {k: v} where v looks call-like: return {name: k, arguments: recurse(v)}
    - Else: return obj unchanged (leaf arguments/value)
    """  # noqa: E501
    if isinstance(obj, dict):
        if "name" in obj and "arguments" in obj:
            args = obj.get("arguments")
            # Only recurse into arguments if it looks like another call
            if _is_call_like(args):
                return {"name": obj.get("name"), "arguments": _to_call_dict(args)}
            return {"name": obj.get("name"), "arguments": args}
        if len(obj) == 1:
            k, v = next(iter(obj.items()))
            # Only convert single-key dicts if the value looks like it could be a call
            if isinstance(v, dict) and _is_call_like(v):
                return {"name": k, "arguments": _to_call_dict(v)}
            # Otherwise, leave it as-is (this is the innermost arguments dict)
            return obj
    return obj


def normalize_to_tool_call_dict(value: Any) -> Any:
    """
    Convert shorthand or nested forms into a direct tool call dict:
    {"name": final_name, "arguments": final_arguments}
    Lists are normalized element-wise.
    """
    if value is None:
        return value

    def _normalize_one(item: Any) -> Any:
        call = _to_call_dict(item)
        return call

    if isinstance(value, list):
        return [_normalize_one(x) for x in value]

    if isinstance(value, dict):
        return _normalize_one(value)

    return value
