"""Tests for deprecation utilities."""

from __future__ import annotations

import logging
import warnings
from typing import Any

from hud.utils.deprecation import deprecated, emit_deprecation_warning


class TestDeprecatedDecorator:
    """Test deprecated decorator."""

    def test_deprecated_function_basic(self, caplog):
        """Test basic function deprecation."""

        @deprecated(reason="This is old")
        def old_function():
            return "result"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with caplog.at_level(logging.WARNING):
                result = old_function()

        # Check function still works
        assert result == "result"

        # Check warning was emitted
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "old_function is deprecated" in str(w[0].message)
        assert "This is old" in str(w[0].message)

        # Check logging
        assert any("old_function is deprecated" in record.message for record in caplog.records)

    def test_deprecated_function_full_params(self):
        """Test function deprecation with all parameters."""

        @deprecated(
            reason="Use new_function instead",
            version="1.0.0",
            replacement="module.new_function",
            removal_version="2.0.0",
        )
        def old_function(x: int) -> int:
            """Old function docstring."""
            return x * 2

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = old_function(5)

        assert result == 10

        message = str(w[0].message)
        assert "deprecated since v1.0.0" in message
        assert "Use module.new_function instead" in message
        assert "Will be removed in v2.0.0" in message

        # Check docstring was updated
        assert old_function.__doc__ is not None
        assert "**DEPRECATED**" in old_function.__doc__
        assert "Old function docstring" in old_function.__doc__

    def test_deprecated_class_basic(self, caplog):
        """Test basic class deprecation."""

        @deprecated(reason="This class is old")
        class OldClass:
            def __init__(self, value: int):
                self.value = value

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with caplog.at_level(logging.WARNING):
                obj = OldClass(42)

        # Check class still works
        assert obj.value == 42

        # Check warning was emitted
        assert len(w) == 1
        assert "OldClass is deprecated" in str(w[0].message)
        assert "This class is old" in str(w[0].message)

        # Check logging
        assert any("OldClass is deprecated" in record.message for record in caplog.records)

    def test_deprecated_class_with_docstring(self):
        """Test class deprecation preserves docstring."""

        @deprecated(reason="Use NewClass", replacement="NewClass")
        class OldClass:
            """Original class docstring."""

        assert OldClass.__doc__ is not None
        assert "**DEPRECATED**" in OldClass.__doc__
        assert "Use NewClass" in OldClass.__doc__
        assert "Original class docstring" in OldClass.__doc__

    def test_deprecated_method(self):
        """Test deprecating instance methods."""

        class MyClass:
            @deprecated(reason="Use new_method")
            def old_method(self) -> str:
                return "old"

        obj = MyClass()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = obj.old_method()

        assert result == "old"
        assert len(w) == 1
        assert "old_method is deprecated" in str(w[0].message)

    def test_deprecated_with_args_kwargs(self):
        """Test deprecated function with various arguments."""

        @deprecated(reason="Testing args")
        def complex_function(
            a: int, b: str, *args: Any, c: bool = True, **kwargs: Any
        ) -> dict[str, Any]:
            return {"a": a, "b": b, "args": args, "c": c, "kwargs": kwargs}

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = complex_function(1, "test", "extra", c=False, key="value")

        assert result == {
            "a": 1,
            "b": "test",
            "args": ("extra",),
            "c": False,
            "kwargs": {"key": "value"},
        }

    def test_deprecated_no_docstring(self):
        """Test deprecated function without existing docstring."""

        @deprecated(reason="No docs")
        def no_doc_func():
            pass

        assert no_doc_func.__doc__ is not None
        assert "**DEPRECATED**" in no_doc_func.__doc__
        assert "No docs" in no_doc_func.__doc__

    def test_deprecated_stacklevel(self):
        """Test that stacklevel points to the right location."""

        @deprecated(reason="Testing stacklevel")
        def deprecated_func():
            pass

        # Capture the warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # This line should be reported as the source
            deprecated_func()

        # The warning should point to this test file, not the decorator
        assert len(w) == 1
        assert "test_deprecation.py" in w[0].filename


class TestEmitDeprecationWarning:
    """Test emit_deprecation_warning function."""

    def test_emit_basic_warning(self, caplog):
        """Test basic deprecation warning emission."""

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with caplog.at_level(logging.WARNING):
                emit_deprecation_warning("This is deprecated")

        # Check warning
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert str(w[0].message) == "This is deprecated"

        # Check logging
        assert any("This is deprecated" in record.message for record in caplog.records)

    def test_emit_custom_category(self):
        """Test emission with custom warning category."""

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            emit_deprecation_warning("Custom warning", category=FutureWarning)

        assert len(w) == 1
        assert issubclass(w[0].category, FutureWarning)

    def test_emit_custom_stacklevel(self):
        """Test emission with custom stack level."""

        def intermediate_function():
            emit_deprecation_warning("From intermediate", stacklevel=3)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            intermediate_function()

        # Should point to this test function, not intermediate_function
        assert len(w) == 1
        assert "test_emit_custom_stacklevel" in w[0].filename or "test_deprecation" in w[0].filename
