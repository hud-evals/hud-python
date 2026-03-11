"""Integration tests for per-session scenario isolation.

These tests run scenarios through the actual MCP protocol via FastMCPClient
to verify that session IDs flow correctly through the full lifecycle:
prompt_handler → _hud_submit → resource_handler.

The key bug these tests guard against: if session IDs don't match across
these three MCP calls, multi-client scenarios break.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

import pytest
from fastmcp.client import Client as FastMCPClient

from hud.environment import Environment


@pytest.fixture()
def env_with_scenarios() -> Environment:
    """Environment with scenarios for testing session isolation."""
    env = Environment("session-test")

    @env.scenario()
    async def greet(name: str) -> AsyncGenerator[Any, Any]:
        answer = yield f"Hello {name}"
        yield 1.0 if answer == "correct" else 0.0

    @env.scenario()
    async def echo(message: str) -> AsyncGenerator[Any, Any]:
        _ = yield message
        yield 1.0

    return env


class TestMCPSessionLifecycle:
    """Test the full prompt → submit → resource lifecycle via MCP client.

    This is the real integration test: an external MCP client connects,
    calls get_prompt, then _hud_submit, then read_resource. The session
    must be consistent across all three calls.
    """

    @pytest.mark.asyncio()
    async def test_full_lifecycle_via_mcp_prompt_and_submit(
        self, env_with_scenarios: Environment
    ) -> None:
        """get_prompt → _hud_submit works via MCP client."""
        async with FastMCPClient(env_with_scenarios) as client:
            prompt = await client.get_prompt("session-test:greet", {"name": "world"})
            assert prompt.messages
            assert "Hello world" in prompt.messages[0].content.text  # type: ignore[union-attr]

            await client.call_tool("_hud_submit", {"scenario": "greet", "answer": "correct"})

            # Session should exist under a real session_id (not __client__)
            # and the answer should be stored
            sessions = env_with_scenarios._scenario_sessions
            assert len(sessions) == 1, (
                f"Expected 1 session, got {len(sessions)}: {list(sessions.keys())}"
            )
            session = next(iter(sessions.values()))
            assert session.answer == "correct"
            assert session.local_name == "greet"

    @pytest.mark.asyncio()
    async def test_two_clients_isolated(self, env_with_scenarios: Environment) -> None:
        """Two separate clients should get isolated scenario sessions."""
        # Client 1 sets up
        await env_with_scenarios.run_scenario_setup(
            "greet", {"name": "alice"}, session_id="client-1"
        )
        # Client 2 sets up same scenario
        await env_with_scenarios.run_scenario_setup("greet", {"name": "bob"}, session_id="client-2")

        # Submit different answers
        await env_with_scenarios.submit("greet", "correct", session_id="client-1")
        await env_with_scenarios.submit("greet", "wrong", session_id="client-2")

        # Evaluate independently
        r1 = await env_with_scenarios.run_scenario_evaluate("greet", session_id="client-1")
        r2 = await env_with_scenarios.run_scenario_evaluate("greet", session_id="client-2")

        assert r1.reward == 1.0
        assert r2.reward == 0.0


class TestSessionEdgeCases:
    """Edge cases that should be handled gracefully."""

    @pytest.mark.asyncio()
    async def test_submit_without_setup_raises(self, env_with_scenarios: Environment) -> None:
        """Submitting without setup should raise, not silently corrupt state."""
        with pytest.raises(ValueError, match="No active"):
            await env_with_scenarios.submit("greet", "answer", session_id="nonexistent")

    @pytest.mark.asyncio()
    async def test_evaluate_without_submit_uses_none_answer(
        self, env_with_scenarios: Environment
    ) -> None:
        """Evaluating without submitting should still work (answer is None)."""
        await env_with_scenarios.run_scenario_setup("echo", {"message": "test"}, session_id="s1")
        # Don't submit -- answer stays None
        result = await env_with_scenarios.run_scenario_evaluate("echo", session_id="s1")
        assert result.reward == 1.0

    @pytest.mark.asyncio()
    async def test_double_evaluate_raises(self, env_with_scenarios: Environment) -> None:
        """Evaluating the same session twice should fail (session is consumed)."""
        await env_with_scenarios.run_scenario_setup("greet", {"name": "x"}, session_id="s1")
        env_with_scenarios._get_session("s1").answer = "correct"  # type: ignore[union-attr]

        await env_with_scenarios.run_scenario_evaluate("greet", session_id="s1")

        with pytest.raises(ValueError, match="No active"):
            await env_with_scenarios.run_scenario_evaluate("greet", session_id="s1")

    @pytest.mark.asyncio()
    async def test_session_cleanup_on_disconnect(self, env_with_scenarios: Environment) -> None:
        """Sessions should be cleaned up when env disconnects."""
        await env_with_scenarios.run_scenario_setup("greet", {"name": "x"}, session_id="s1")
        assert env_with_scenarios._get_session("s1") is not None

        # Simulate disconnect cleanup
        env_with_scenarios._scenario_sessions = {}
        assert env_with_scenarios._get_session("s1") is None

    @pytest.mark.asyncio()
    async def test_scenario_mismatch_raises(self, env_with_scenarios: Environment) -> None:
        """Submitting to wrong scenario name raises."""
        await env_with_scenarios.run_scenario_setup("greet", {"name": "x"}, session_id="s1")
        with pytest.raises(ValueError, match="Scenario mismatch"):
            await env_with_scenarios.submit("echo", "answer", session_id="s1")


class TestBackwardCompat:
    """Ensure the __client__ default key still works for non-MCP usage."""

    @pytest.mark.asyncio()
    async def test_no_session_id_uses_default_key(self, env_with_scenarios: Environment) -> None:
        """When no session_id is passed, uses __client__ default."""
        prompt = await env_with_scenarios.run_scenario_setup("greet", {"name": "world"})
        assert prompt == "Hello world"
        assert env_with_scenarios._active_session is not None
        assert env_with_scenarios._active_session.local_name == "greet"

    @pytest.mark.asyncio()
    async def test_full_lifecycle_without_session_id(self, env_with_scenarios: Environment) -> None:
        """Complete lifecycle without session_id (backward compat path)."""
        await env_with_scenarios.run_scenario_setup("greet", {"name": "x"})
        await env_with_scenarios.submit("greet", "correct")
        result = await env_with_scenarios.run_scenario_evaluate("greet")
        assert result.reward == 1.0
