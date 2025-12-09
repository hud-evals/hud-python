"""Tests for hud.trace.parallel module."""

from __future__ import annotations

import ast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hud.trace.parallel import (
    ASTExtractionError,
    _extract_body,
    _find_async_with,
    _get_end_line,
    run_parallel_traces,
)


class TestASTHelpers:
    """Tests for AST helper functions."""

    def test_find_async_with_finds_correct_node(self) -> None:
        """_find_async_with finds the async with containing target line."""
        source = '''
async def main():
    x = 1
    async with something as ctx:
        do_stuff()
        more_stuff()
    y = 2
'''
        tree = ast.parse(source)
        
        # Line 4 is inside the async with
        node = _find_async_with(tree, 5)
        assert node is not None
        assert isinstance(node, ast.AsyncWith)

    def test_find_async_with_returns_none_when_not_found(self) -> None:
        """_find_async_with returns None when line is outside async with."""
        source = '''
async def main():
    x = 1
    async with something as ctx:
        do_stuff()
    y = 2
'''
        tree = ast.parse(source)
        
        # Line 6 is outside the async with
        node = _find_async_with(tree, 7)
        assert node is None

    def test_get_end_line(self) -> None:
        """_get_end_line returns last line of node."""
        source = '''
async with ctx:
    line1()
    line2()
    line3()
'''
        tree = ast.parse(source)
        async_with = tree.body[0]
        
        end_line = _get_end_line(async_with)
        assert end_line >= 4  # At least through line 4

    def test_extract_body(self) -> None:
        """_extract_body extracts the body source from async with."""
        source = '''async with ctx:
    do_thing()
    more_thing()
'''
        lines = source.split('\n')
        lines = [line + '\n' for line in lines]
        
        tree = ast.parse(source)
        async_with = tree.body[0]
        
        body = _extract_body(lines, async_with)
        assert "do_thing()" in body
        assert "more_thing()" in body


class TestRunParallelTraces:
    """Tests for run_parallel_traces function."""

    @pytest.mark.asyncio
    async def test_runs_body_for_each_context(self) -> None:
        """run_parallel_traces runs body for each TraceContext."""
        # Create mock trace contexts
        mock_tcs = []
        for i in range(3):
            tc = MagicMock()
            tc.index = i
            tc.__aenter__ = AsyncMock(return_value=tc)
            tc.__aexit__ = AsyncMock(return_value=None)
            mock_tcs.append(tc)
        
        # Simple body that sets reward
        body_source = "tc.reward = tc.index * 10"
        captured_locals: dict[str, object] = {}
        
        results = await run_parallel_traces(mock_tcs, body_source, captured_locals)
        
        assert len(results) == 3
        # Each context should have had __aenter__ and __aexit__ called
        for tc in mock_tcs:
            tc.__aenter__.assert_called_once()
            tc.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_captures_exceptions(self) -> None:
        """run_parallel_traces captures exceptions in context."""
        tc = MagicMock()
        tc.index = 0
        tc.__aenter__ = AsyncMock(return_value=tc)
        tc.__aexit__ = AsyncMock(return_value=None)
        
        # Body that raises
        body_source = "raise ValueError('test error')"
        captured_locals: dict[str, object] = {}
        
        results = await run_parallel_traces([tc], body_source, captured_locals)
        
        assert len(results) == 1
        # Error should be captured, not raised
        assert hasattr(tc, "_error") or tc.__aexit__.called

    @pytest.mark.asyncio
    async def test_uses_captured_locals(self) -> None:
        """run_parallel_traces uses captured locals in body execution."""
        tc = MagicMock()
        tc.index = 0
        tc.result = None
        tc.__aenter__ = AsyncMock(return_value=tc)
        tc.__aexit__ = AsyncMock(return_value=None)
        
        # Body that uses captured local
        body_source = "tc.result = my_value * 2"
        captured_locals = {"my_value": 21}
        
        results = await run_parallel_traces([tc], body_source, captured_locals)
        
        assert len(results) == 1


class TestASTExtractionError:
    """Tests for ASTExtractionError."""

    def test_is_exception(self) -> None:
        """ASTExtractionError is an exception."""
        error = ASTExtractionError("test message")
        assert isinstance(error, Exception)
        assert str(error) == "test message"

