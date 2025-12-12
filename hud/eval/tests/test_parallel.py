"""Tests for hud.eval.parallel module."""

from __future__ import annotations

import ast

import pytest

from hud.eval.parallel import (
    ASTExtractionError,
    _extract_body,
    _find_async_with,
    _get_end_line,
    expand_variants,
    resolve_group_ids,
)


class TestExpandVariants:
    """Tests for expand_variants helper."""

    def test_none_returns_empty_dict(self) -> None:
        """None variants returns list with empty dict."""
        result = expand_variants(None)
        assert result == [{}]

    def test_empty_dict_returns_empty_dict(self) -> None:
        """Empty variants returns list with empty dict."""
        result = expand_variants({})
        assert result == [{}]

    def test_single_value_stays_single(self) -> None:
        """Single non-list value stays as single variant."""
        result = expand_variants({"model": "gpt-4o"})
        assert result == [{"model": "gpt-4o"}]

    def test_list_expands_to_variants(self) -> None:
        """List value expands to multiple variants."""
        result = expand_variants({"model": ["gpt-4o", "claude"]})
        assert result == [{"model": "gpt-4o"}, {"model": "claude"}]

    def test_multiple_lists_create_combinations(self) -> None:
        """Multiple lists create all combinations."""
        result = expand_variants(
            {
                "model": ["a", "b"],
                "temp": [0.0, 1.0],
            }
        )

        assert len(result) == 4
        assert {"model": "a", "temp": 0.0} in result
        assert {"model": "a", "temp": 1.0} in result
        assert {"model": "b", "temp": 0.0} in result
        assert {"model": "b", "temp": 1.0} in result

    def test_mixed_single_and_list(self) -> None:
        """Mixed single values and lists work correctly."""
        result = expand_variants(
            {
                "model": ["gpt-4o", "claude"],
                "temp": 0.7,
            }
        )

        assert len(result) == 2
        assert {"model": "gpt-4o", "temp": 0.7} in result
        assert {"model": "claude", "temp": 0.7} in result


class TestResolveGroupIds:
    """Tests for resolve_group_ids helper."""

    def test_uses_provided_group_ids(self) -> None:
        """Uses provided group_ids when given."""
        result = resolve_group_ids(["a", "b", "c"], 3)
        assert result == ["a", "b", "c"]

    def test_generates_shared_group_id(self) -> None:
        """Generates shared group_id when not provided."""
        result = resolve_group_ids(None, 3)
        assert len(result) == 3
        # All should be the same
        assert result[0] == result[1] == result[2]
        # Should be a valid UUID
        assert len(result[0]) == 36

    def test_raises_on_length_mismatch(self) -> None:
        """Raises ValueError when group_ids length doesn't match."""
        with pytest.raises(ValueError, match="group_ids length"):
            resolve_group_ids(["a", "b"], 3)


class TestASTHelpers:
    """Tests for AST helper functions."""

    def test_find_async_with_finds_correct_node(self) -> None:
        """_find_async_with finds the async with containing target line."""
        source = """
async def main():
    x = 1
    async with something as ctx:
        do_stuff()
        more_stuff()
    y = 2
"""
        tree = ast.parse(source)

        # Line 5 is inside the async with
        node = _find_async_with(tree, 5)
        assert node is not None
        assert isinstance(node, ast.AsyncWith)

    def test_find_async_with_returns_none_when_not_found(self) -> None:
        """_find_async_with returns None when line is outside async with."""
        source = """
async def main():
    x = 1
    async with something as ctx:
        do_stuff()
    y = 2
"""
        tree = ast.parse(source)

        # Line 7 is outside the async with
        node = _find_async_with(tree, 7)
        assert node is None

    def test_get_end_line(self) -> None:
        """_get_end_line returns last line of node."""
        source = """
async with ctx:
    line1()
    line2()
    line3()
"""
        tree = ast.parse(source)
        async_with = tree.body[0]

        end_line = _get_end_line(async_with)
        assert end_line >= 4  # At least through line 4

    def test_extract_body(self) -> None:
        """_extract_body extracts the body source from async with."""
        source = """async with ctx:
    do_thing()
    more_thing()
"""
        lines = source.split("\n")
        lines = [line + "\n" for line in lines]

        tree = ast.parse(source)
        async_with = tree.body[0]
        assert isinstance(async_with, ast.AsyncWith)

        body = _extract_body(lines, async_with)
        assert "do_thing()" in body
        assert "more_thing()" in body


class TestASTExtractionError:
    """Tests for ASTExtractionError."""

    def test_is_exception(self) -> None:
        """ASTExtractionError is an exception."""
        error = ASTExtractionError("test message")
        assert isinstance(error, Exception)
        assert str(error) == "test message"
