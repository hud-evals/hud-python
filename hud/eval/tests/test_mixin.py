"""Tests for hud.eval.mixin module."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from hud.eval.mixin import EvalMixin
from hud.eval.parallel import expand_variants


class TestExpandVariants:
    """Tests for expand_variants helper."""

    def test_none_returns_empty_dict(self) -> None:
        result = expand_variants(None)
        assert result == [{}]

    def test_single_value_stays_single(self) -> None:
        result = expand_variants({"model": "gpt-4o"})
        assert result == [{"model": "gpt-4o"}]

    def test_list_expands_to_variants(self) -> None:
        result = expand_variants({"model": ["gpt-4o", "claude"]})
        assert result == [{"model": "gpt-4o"}, {"model": "claude"}]

    def test_multiple_lists_create_combinations(self) -> None:
        result = expand_variants({"model": ["a", "b"], "temp": [0.0, 1.0]})
        assert len(result) == 4
        assert {"model": "a", "temp": 0.0} in result
        assert {"model": "b", "temp": 1.0} in result


class MockEnvironment(EvalMixin):
    """Mock environment for testing EvalMixin."""

    def __init__(self) -> None:
        self.name = "test-env"
        self._connections: dict[str, Any] = {}
        self._last_evals = None
        self._hub_configs: list[dict[str, Any]] = []
        self._setup_calls: list[tuple[str, dict[str, Any]]] = []
        self._evaluate_calls: list[tuple[str, dict[str, Any]]] = []
        self.prompt: str | None = None

    @property
    def is_parallelizable(self) -> bool:
        return all(getattr(c, "is_remote", True) for c in self._connections.values())

    @property
    def local_connections(self) -> list[str]:
        return [name for name, c in self._connections.items() if getattr(c, "is_local", False)]


class TestEvalMixin:
    """Tests for EvalMixin."""

    @pytest.mark.asyncio
    async def test_eval_single_creates_context(self) -> None:
        """eval() with group=1 creates single EvalContext."""
        env = MockEnvironment()

        async with env.eval("test-task") as ctx:
            assert ctx.eval_name == "test-task"
            assert ctx.trace_id is not None
            assert ctx.variants == {}

    @pytest.mark.asyncio
    async def test_eval_sets_reward(self) -> None:
        """reward can be set on EvalContext."""
        env = MockEnvironment()

        async with env.eval("test-task") as ctx:
            ctx.reward = 0.95

        assert ctx.reward == 0.95

    @pytest.mark.asyncio
    async def test_eval_with_variants_single(self) -> None:
        """eval() with single variant value works."""
        env = MockEnvironment()

        async with env.eval("test-task", variants={"model": "gpt-4o"}) as ctx:
            assert ctx.variants == {"model": "gpt-4o"}

    @pytest.mark.asyncio
    async def test_eval_rejects_parallel_with_local_connections(self) -> None:
        """eval() raises error for parallel with local connections."""
        env = MockEnvironment()

        # Add a local connection
        mock_conn = MagicMock()
        mock_conn.is_local = True
        mock_conn.is_remote = False
        env._connections["local-server"] = mock_conn

        with pytest.raises(ValueError, match="Cannot run parallel evals"):
            async with env.eval("test-task", group=2) as _ctx:
                pass

    @pytest.mark.asyncio
    async def test_eval_allows_parallel_with_remote_connections(self) -> None:
        """eval() allows parallel with only remote connections."""
        env = MockEnvironment()

        # Add a remote connection
        mock_conn = MagicMock()
        mock_conn.is_local = False
        mock_conn.is_remote = True
        env._connections["remote-server"] = mock_conn

        # Just verify it doesn't raise the local connection error
        assert env.is_parallelizable is True

    @pytest.mark.asyncio
    async def test_eval_rejects_zero_group(self) -> None:
        """eval() raises error for group <= 0."""
        env = MockEnvironment()

        with pytest.raises(ValueError, match="group must be >= 1"):
            async with env.eval("test-task", group=0) as _ctx:
                pass

    def test_last_evals_none_initially(self) -> None:
        """last_evals is None before any parallel execution."""
        env = MockEnvironment()
        assert env.last_evals is None
