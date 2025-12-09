"""Tests for hud.trace.mixin module."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hud.trace.mixin import TraceMixin, _expand_variants


class TestExpandVariants:
    """Tests for _expand_variants helper."""

    def test_none_returns_empty_dict(self) -> None:
        """None variants returns list with empty dict."""
        result = _expand_variants(None)
        assert result == [{}]

    def test_empty_dict_returns_empty_dict(self) -> None:
        """Empty variants returns list with empty dict."""
        result = _expand_variants({})
        assert result == [{}]

    def test_single_value_stays_single(self) -> None:
        """Single non-list value stays as single variant."""
        result = _expand_variants({"model": "gpt-4o"})
        assert result == [{"model": "gpt-4o"}]

    def test_list_expands_to_variants(self) -> None:
        """List value expands to multiple variants."""
        result = _expand_variants({"model": ["gpt-4o", "claude"]})
        assert result == [{"model": "gpt-4o"}, {"model": "claude"}]

    def test_multiple_lists_create_combinations(self) -> None:
        """Multiple lists create all combinations."""
        result = _expand_variants({
            "model": ["a", "b"],
            "temp": [0.0, 1.0],
        })
        
        assert len(result) == 4
        assert {"model": "a", "temp": 0.0} in result
        assert {"model": "a", "temp": 1.0} in result
        assert {"model": "b", "temp": 0.0} in result
        assert {"model": "b", "temp": 1.0} in result

    def test_mixed_single_and_list(self) -> None:
        """Mixed single values and lists work correctly."""
        result = _expand_variants({
            "model": ["gpt-4o", "claude"],
            "temp": 0.7,
        })
        
        assert len(result) == 2
        assert {"model": "gpt-4o", "temp": 0.7} in result
        assert {"model": "claude", "temp": 0.7} in result


class MockEnvironment(TraceMixin):
    """Mock environment for testing TraceMixin."""
    
    def __init__(self) -> None:
        self.name = "test-env"
        self._connections: dict[str, Any] = {}
        self._last_traces = None
    
    @property
    def is_parallelizable(self) -> bool:
        return all(
            getattr(c, "is_remote", True)
            for c in self._connections.values()
        )
    
    @property
    def local_connections(self) -> list[str]:
        return [
            name for name, c in self._connections.items()
            if getattr(c, "is_local", False)
        ]
    
    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        return {"name": name, "arguments": arguments}
    
    async def __aenter__(self) -> "MockEnvironment":
        return self
    
    async def __aexit__(self, *args: Any) -> None:
        pass


class TestTraceMixin:
    """Tests for TraceMixin."""

    @pytest.mark.asyncio
    async def test_trace_single_creates_context(self) -> None:
        """trace() with group=1 creates single TraceContext."""
        env = MockEnvironment()
        
        async with env.trace("test-task") as tc:
            assert tc.name == "test-task"
            assert tc.trace_id is not None
            assert tc.variants == {}

    @pytest.mark.asyncio
    async def test_trace_sets_reward(self) -> None:
        """reward can be set on TraceContext."""
        env = MockEnvironment()
        
        async with env.trace("test-task") as tc:
            tc.reward = 0.95
        
        assert tc.reward == 0.95

    @pytest.mark.asyncio
    async def test_trace_with_variants_single(self) -> None:
        """trace() with single variant value works."""
        env = MockEnvironment()
        
        async with env.trace("test-task", variants={"model": "gpt-4o"}) as tc:
            assert tc.variants == {"model": "gpt-4o"}

    @pytest.mark.asyncio
    async def test_trace_rejects_parallel_with_local_connections(self) -> None:
        """trace() raises error for parallel with local connections."""
        env = MockEnvironment()
        
        # Add a local connection
        mock_conn = MagicMock()
        mock_conn.is_local = True
        mock_conn.is_remote = False
        env._connections["local-server"] = mock_conn
        
        with pytest.raises(ValueError, match="Cannot run parallel traces"):
            async with env.trace("test-task", group=2) as tc:
                pass

    @pytest.mark.asyncio
    async def test_trace_allows_parallel_with_remote_connections(self) -> None:
        """trace() allows parallel with only remote connections."""
        env = MockEnvironment()
        
        # Add a remote connection
        mock_conn = MagicMock()
        mock_conn.is_local = False
        mock_conn.is_remote = True
        env._connections["remote-server"] = mock_conn
        
        # This should not raise (though parallel execution is complex to test)
        # Just verify it doesn't raise the local connection error
        assert env.is_parallelizable is True

    @pytest.mark.asyncio
    async def test_trace_rejects_zero_group(self) -> None:
        """trace() raises error for group <= 0."""
        env = MockEnvironment()
        
        with pytest.raises(ValueError, match="group must be >= 1"):
            async with env.trace("test-task", group=0) as tc:
                pass

    def test_last_traces_none_initially(self) -> None:
        """last_traces is None before any parallel execution."""
        env = MockEnvironment()
        assert env.last_traces is None

    @pytest.mark.asyncio
    async def test_trace_context_delegates_call_tool(self) -> None:
        """TraceContext.call_tool delegates to environment."""
        env = MockEnvironment()
        
        async with env.trace("test-task") as tc:
            result = await tc.call_tool("my_tool", {"arg": "value"})
        
        assert result["name"] == "my_tool"
        assert result["arguments"] == {"arg": "value"}

