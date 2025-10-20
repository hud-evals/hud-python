from __future__ import annotations

from hud.utils.tool_shorthand import (
    _is_call_like,
    _to_call_dict,
    normalize_to_tool_call_dict,
)


def test_is_call_like_with_name_and_arguments():
    """Test _is_call_like with name and arguments keys."""
    obj = {"name": "test_tool", "arguments": {"key": "value"}}
    assert _is_call_like(obj) is True


def test_is_call_like_with_single_key_dict_value():
    """Test _is_call_like with single key dict containing dict value."""
    obj = {"tool": {"name": "test"}}
    assert _is_call_like(obj) is True


def test_is_call_like_with_nested_single_key():
    """Test _is_call_like with nested single key dict."""
    obj = {"tool": {"inner": {"key": "value"}}}
    assert _is_call_like(obj) is True


def test_is_call_like_not_dict():
    """Test _is_call_like returns False for non-dict."""
    assert _is_call_like("string") is False
    assert _is_call_like(123) is False
    assert _is_call_like(None) is False
    assert _is_call_like([]) is False


def test_is_call_like_empty_dict():
    """Test _is_call_like returns False for empty dict."""
    assert _is_call_like({}) is False


def test_is_call_like_multi_key_dict():
    """Test _is_call_like returns False for multi-key dict without name/arguments."""
    obj = {"key1": "value1", "key2": "value2"}
    assert _is_call_like(obj) is False


def test_to_call_dict_with_name_arguments():
    """Test _to_call_dict preserves name and arguments."""
    obj = {"name": "test_tool", "arguments": {"param": "value"}}
    result = _to_call_dict(obj)
    assert result == {"name": "test_tool", "arguments": {"param": "value"}}


def test_to_call_dict_with_nested_call():
    """Test _to_call_dict with nested call-like arguments."""
    obj = {"name": "outer", "arguments": {"name": "inner", "arguments": {"x": 1}}}
    result = _to_call_dict(obj)
    assert result == {"name": "outer", "arguments": {"name": "inner", "arguments": {"x": 1}}}


def test_to_call_dict_shorthand_single_key():
    """Test _to_call_dict converts shorthand single-key dict."""
    obj = {"tool_name": {"name": "inner", "arguments": {}}}
    result = _to_call_dict(obj)
    assert result == {"name": "tool_name", "arguments": {"name": "inner", "arguments": {}}}


def test_to_call_dict_non_call_arguments():
    """Test _to_call_dict with non-call-like arguments."""
    obj = {"name": "test", "arguments": {"simple": "value"}}
    result = _to_call_dict(obj)
    assert result == {"name": "test", "arguments": {"simple": "value"}}


def test_to_call_dict_non_dict():
    """Test _to_call_dict returns non-dict unchanged."""
    assert _to_call_dict("string") == "string"
    assert _to_call_dict(123) == 123
    assert _to_call_dict(None) is None


def test_to_call_dict_single_key_non_call():
    """Test _to_call_dict with single key but non-call value."""
    obj = {"key": "simple_value"}
    result = _to_call_dict(obj)
    assert result == {"key": "simple_value"}


def test_normalize_to_tool_call_dict_none():
    """Test normalize_to_tool_call_dict with None."""
    assert normalize_to_tool_call_dict(None) is None


def test_normalize_to_tool_call_dict_simple_dict():
    """Test normalize_to_tool_call_dict with simple dict."""
    obj = {"name": "tool", "arguments": {"x": 1}}
    result = normalize_to_tool_call_dict(obj)
    assert result == {"name": "tool", "arguments": {"x": 1}}


def test_normalize_to_tool_call_dict_shorthand():
    """Test normalize_to_tool_call_dict with shorthand notation."""
    obj = {"tool_name": {"name": "inner", "arguments": {}}}
    result = normalize_to_tool_call_dict(obj)
    assert result == {"name": "tool_name", "arguments": {"name": "inner", "arguments": {}}}


def test_normalize_to_tool_call_dict_list():
    """Test normalize_to_tool_call_dict with list of dicts."""
    obj = [
        {"name": "tool1", "arguments": {"a": 1}},
        {"name": "tool2", "arguments": {"b": 2}},
    ]
    result = normalize_to_tool_call_dict(obj)
    assert len(result) == 2
    assert result[0] == {"name": "tool1", "arguments": {"a": 1}}
    assert result[1] == {"name": "tool2", "arguments": {"b": 2}}


def test_normalize_to_tool_call_dict_list_shorthand():
    """Test normalize_to_tool_call_dict with list of shorthand dicts."""
    obj = [
        {"tool1": {"name": "inner1", "arguments": {}}},
        {"tool2": {"name": "inner2", "arguments": {}}},
    ]
    result = normalize_to_tool_call_dict(obj)
    assert len(result) == 2
    assert result[0]["name"] == "tool1"
    assert result[1]["name"] == "tool2"


def test_normalize_to_tool_call_dict_non_dict_non_list():
    """Test normalize_to_tool_call_dict with non-dict, non-list value."""
    assert normalize_to_tool_call_dict("string") == "string"
    assert normalize_to_tool_call_dict(123) == 123


def test_normalize_to_tool_call_dict_empty_list():
    """Test normalize_to_tool_call_dict with empty list."""
    assert normalize_to_tool_call_dict([]) == []


def test_normalize_to_tool_call_dict_complex_nested():
    """Test normalize_to_tool_call_dict with complex nested structure."""
    obj = {
        "outer_tool": {
            "name": "middle_tool",
            "arguments": {"name": "inner_tool", "arguments": {"x": 1}},
        }
    }
    result = normalize_to_tool_call_dict(obj)
    assert result["name"] == "outer_tool"
    assert result["arguments"]["name"] == "middle_tool"
    assert result["arguments"]["arguments"]["name"] == "inner_tool"
