from __future__ import annotations

from dataclasses import dataclass

from hud.utils.serialization import json_safe_dict, json_safe_value


def test_json_safe_value_serializes_dataclass() -> None:
    @dataclass
    class Demo:
        name: str
        count: int

    result = json_safe_value(Demo(name="test", count=2))
    assert result == {"name": "test", "count": 2}


def test_json_safe_value_falls_back_for_unserializable_object() -> None:
    class Weird:
        def __init__(self) -> None:
            raise RuntimeError("boom")

    value = Weird.__new__(Weird)
    result = json_safe_value(value)
    assert isinstance(result, str)
    assert "Weird" in result


def test_json_safe_dict_serializes_each_value() -> None:
    data = {"number": 1, "items": [1, 2, 3]}
    assert json_safe_dict(data) == data
