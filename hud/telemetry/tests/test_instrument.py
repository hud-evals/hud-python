from __future__ import annotations

from dataclasses import dataclass

import pytest
from opentelemetry.trace import SpanKind

from hud.telemetry.instrument import _serialize_value, instrument


def test_serialize_value_simple_types():
    """Test _serialize_value with simple types."""
    assert _serialize_value("string") == "string"
    assert _serialize_value(42) == 42
    assert _serialize_value(3.14) == 3.14
    assert _serialize_value(True) is True
    assert _serialize_value(None) is None


def test_serialize_value_list():
    """Test _serialize_value with lists."""
    result = _serialize_value([1, 2, 3])
    assert result == [1, 2, 3]


def test_serialize_value_list_truncation():
    """Test _serialize_value truncates long lists."""
    long_list = list(range(20))
    result = _serialize_value(long_list, max_items=5)
    assert len(result) == 5
    assert result == [0, 1, 2, 3, 4]


def test_serialize_value_tuple():
    """Test _serialize_value with tuples."""
    result = _serialize_value((1, 2, 3))
    assert result == [1, 2, 3]  # Converted to list by JSON


def test_serialize_value_tuple_truncation():
    """Test _serialize_value truncates long tuples."""
    long_tuple = tuple(range(20))
    result = _serialize_value(long_tuple, max_items=5)
    assert len(result) == 5


def test_serialize_value_dict():
    """Test _serialize_value with dicts."""
    result = _serialize_value({"key": "value"})
    assert result == {"key": "value"}


def test_serialize_value_dict_truncation():
    """Test _serialize_value truncates large dicts."""
    large_dict = {f"key{i}": i for i in range(20)}
    result = _serialize_value(large_dict, max_items=5)
    assert len(result) == 5


def test_serialize_value_complex_object():
    """Test _serialize_value with custom objects."""

    @dataclass
    class CustomObj:
        name: str
        value: int

    obj = CustomObj(name="test", value=42)
    result = _serialize_value(obj)
    assert isinstance(result, dict)
    assert result["name"] == "test"
    assert result["value"] == 42


def test_serialize_value_fallback():
    """Test _serialize_value fallback for non-serializable objects."""

    class WeirdObj:
        def __init__(self):
            raise Exception("Can't access")

    obj = WeirdObj.__new__(WeirdObj)
    result = _serialize_value(obj)
    # The result is a string representation of the object
    assert isinstance(result, str)
    assert "WeirdObj" in result


@pytest.mark.asyncio
async def test_instrument_async_basic():
    """Test instrument decorator on async function."""

    @instrument
    async def test_func(x: int, y: int) -> int:
        return x + y

    result = await test_func(2, 3)
    assert result == 5


@pytest.mark.asyncio
async def test_instrument_async_with_params():
    """Test instrument with custom parameters."""

    @instrument(name="custom_name", span_type="custom_type")
    async def test_func(x: int) -> int:
        return x * 2

    result = await test_func(5)
    assert result == 10


@pytest.mark.asyncio
async def test_instrument_async_with_exception():
    """Test instrument handles exceptions."""

    @instrument
    async def test_func():
        raise ValueError("Test error")

    with pytest.raises(ValueError, match="Test error"):
        await test_func()


@pytest.mark.asyncio
async def test_instrument_async_no_record_args():
    """Test instrument with record_args=False."""

    @instrument(record_args=False)
    async def test_func(x: int) -> int:
        return x

    result = await test_func(42)
    assert result == 42


@pytest.mark.asyncio
async def test_instrument_async_no_record_result():
    """Test instrument with record_result=False."""

    @instrument(record_result=False)
    async def test_func() -> str:
        return "test"

    result = await test_func()
    assert result == "test"


@pytest.mark.asyncio
async def test_instrument_async_with_attributes():
    """Test instrument with custom attributes."""

    @instrument(attributes={"custom_attr": "value"})
    async def test_func() -> int:
        return 42

    result = await test_func()
    assert result == 42


@pytest.mark.asyncio
async def test_instrument_async_with_span_kind():
    """Test instrument with custom span kind."""

    @instrument(span_kind=SpanKind.CLIENT)
    async def test_func() -> int:
        return 1

    result = await test_func()
    assert result == 1


def test_instrument_sync_basic():
    """Test instrument decorator on sync function."""

    @instrument
    def test_func(x: int, y: int) -> int:
        return x + y

    result = test_func(2, 3)
    assert result == 5


def test_instrument_sync_with_params():
    """Test instrument on sync function with parameters."""

    @instrument(name="sync_custom", span_type="sync_type")
    def test_func(x: int) -> int:
        return x * 2

    result = test_func(5)
    assert result == 10


def test_instrument_sync_with_exception():
    """Test instrument handles exceptions in sync functions."""

    @instrument
    def test_func():
        raise ValueError("Sync error")

    with pytest.raises(ValueError, match="Sync error"):
        test_func()


def test_instrument_sync_no_record_args():
    """Test instrument sync with record_args=False."""

    @instrument(record_args=False)
    def test_func(x: int) -> int:
        return x

    result = test_func(42)
    assert result == 42


def test_instrument_sync_no_record_result():
    """Test instrument sync with record_result=False."""

    @instrument(record_result=False)
    def test_func() -> str:
        return "test"

    result = test_func()
    assert result == "test"


def test_instrument_sync_with_attributes():
    """Test instrument sync with custom attributes."""

    @instrument(attributes={"sync_attr": "sync_value"})
    def test_func() -> int:
        return 42

    result = test_func()
    assert result == 42


def test_instrument_already_instrumented():
    """Test that instrumenting already instrumented function is skipped."""

    @instrument
    def test_func():
        return "original"

    # Try to instrument again
    test_func2 = instrument(test_func)

    # Should be the same function
    assert test_func2 is test_func


def test_instrument_marks_as_instrumented():
    """Test that instrument marks functions correctly."""

    @instrument
    def test_func():
        return True

    assert hasattr(test_func, "_hud_instrumented")
    assert test_func._hud_instrumented is True
    assert hasattr(test_func, "_hud_original")


@pytest.mark.asyncio
async def test_instrument_async_complex_result():
    """Test instrument with complex result object."""

    @instrument
    async def test_func() -> dict:
        return {"nested": {"data": [1, 2, 3]}, "count": 3}

    result = await test_func()
    assert result["count"] == 3


def test_instrument_sync_complex_result():
    """Test instrument sync with complex result."""

    @dataclass
    class Result:
        value: int
        name: str

    @instrument
    def test_func() -> Result:
        return Result(value=42, name="test")

    result = test_func()
    assert result.value == 42


@pytest.mark.asyncio
async def test_instrument_async_with_self_param():
    """Test instrument properly handles 'self' parameter."""

    class TestClass:
        @instrument
        async def method(self, x: int) -> int:
            return x * 2

    obj = TestClass()
    result = await obj.method(5)
    assert result == 10


def test_instrument_sync_with_cls_param():
    """Test instrument properly handles 'cls' parameter."""

    class TestClass:
        @classmethod
        @instrument
        def method(cls, x: int) -> int:
            return x * 3

    result = TestClass.method(4)
    assert result == 12


@pytest.mark.asyncio
async def test_instrument_async_serialization_error():
    """Test instrument handles serialization errors gracefully."""

    class UnserializableArg:
        def __getattribute__(self, name):
            raise Exception("Can't serialize")

    @instrument
    async def test_func(arg):
        return "success"

    # Should not raise, just skip serialization
    result = await test_func(UnserializableArg())
    assert result == "success"


def test_instrument_function_without_signature():
    """Test instrument on functions without inspectable signature."""
    # Built-in functions don't have signatures
    instrumented_len = instrument(len)
    result = instrumented_len([1, 2, 3])
    assert result == 3


@pytest.mark.asyncio
async def test_instrument_async_result_serialization_error():
    """Test instrument handles result serialization errors."""

    class UnserializableResult:
        def __iter__(self):
            raise Exception("Can't iterate")

    @instrument
    async def test_func():
        return UnserializableResult()

    # Should not raise, just skip result recording
    result = await test_func()
    assert isinstance(result, UnserializableResult)


def test_instrument_without_parentheses():
    """Test using @instrument without parentheses."""

    @instrument
    def test_func(x: int) -> int:
        return x + 1

    assert test_func(5) == 6


def test_instrument_with_parentheses():
    """Test using @instrument() with parentheses."""

    @instrument()
    def test_func(x: int) -> int:
        return x + 1

    assert test_func(5) == 6


@pytest.mark.asyncio
async def test_instrument_async_with_defaults():
    """Test instrument with function that has default arguments."""

    @instrument
    async def test_func(x: int, y: int = 10) -> int:
        return x + y

    assert await test_func(5) == 15
    assert await test_func(5, 20) == 25


def test_instrument_sync_with_kwargs():
    """Test instrument with keyword arguments."""

    @instrument
    def test_func(x: int, **kwargs) -> dict:
        return {"x": x, **kwargs}

    result = test_func(1, a=2, b=3)
    assert result == {"x": 1, "a": 2, "b": 3}


@pytest.mark.asyncio
async def test_instrument_async_with_varargs():
    """Test instrument with *args."""

    @instrument
    async def test_func(*args) -> int:
        return sum(args)

    result = await test_func(1, 2, 3, 4)
    assert result == 10
