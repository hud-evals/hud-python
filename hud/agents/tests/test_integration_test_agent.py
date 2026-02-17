"""Tests for IntegrationTestRunner."""

from __future__ import annotations

import asyncio

import pytest

from hud.agents.misc import IntegrationTestRunner


def test_runs_all_integration_test_calls(mock_eval_context) -> None:
    """Runner executes each configured integration test call in order."""

    async def _run() -> None:
        mock_eval_context._integration_test_calls = [
            ("tool_a", {"x": 1}),
            ("tool_b", {"y": "ok"}),
        ]

        runner = IntegrationTestRunner.create()
        result = await runner.run(mock_eval_context)

        assert result.done is True
        assert mock_eval_context.tool_calls == [
            ("tool_a", {"x": 1}),
            ("tool_b", {"y": "ok"}),
        ]

    asyncio.run(_run())


def test_raises_when_no_integration_test_calls(mock_eval_context) -> None:
    """Runner fails fast when no integration calls are configured."""

    async def _run() -> None:
        runner = IntegrationTestRunner.create()

        with pytest.raises(ValueError, match="integration_test_tool"):
            await runner.run(mock_eval_context)

    asyncio.run(_run())
