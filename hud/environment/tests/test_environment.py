"""Tests for Environment class - context manager, resources, prompts, prompt feature."""

from __future__ import annotations

import pytest


class TestEnvironmentPrompt:
    """Tests for Environment.prompt feature."""

    def test_prompt_defaults_to_none(self) -> None:
        """Environment.prompt defaults to None."""
        from hud.environment import Environment

        env = Environment("test")
        assert env.prompt is None

    def test_prompt_can_be_set(self) -> None:
        """Environment.prompt can be set manually."""
        from hud.environment import Environment

        env = Environment("test")
        env.prompt = "Navigate to google.com"
        assert env.prompt == "Navigate to google.com"


class TestEnvironmentContextManager:
    """Tests for Environment async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_sets_in_context_flag(self) -> None:
        """Context manager sets _in_context flag."""
        from hud.environment import Environment

        env = Environment("test")

        assert env._in_context is False

        async with env:
            assert env._in_context is True

        assert env._in_context is False

    @pytest.mark.asyncio
    async def test_context_manager_no_connections(self) -> None:
        """Context manager works with no connections."""
        from hud.environment import Environment

        env = Environment("test")

        async with env:
            # Should work without connections
            pass


class TestEnvironmentResources:
    """Tests for Environment resource operations."""

    @pytest.mark.asyncio
    async def test_list_resources_empty(self) -> None:
        """list_resources returns empty list when no resources."""
        from hud.environment import Environment

        env = Environment("test")

        async with env:
            resources = await env.list_resources()

        assert resources == []

    @pytest.mark.asyncio
    async def test_read_resource_not_found(self) -> None:
        """read_resource raises when resource not found."""
        from hud.environment import Environment

        env = Environment("test")

        async with env:
            with pytest.raises(ValueError, match="Resource not found"):
                await env.read_resource("file://nonexistent.txt")


class TestEnvironmentPrompts:
    """Tests for Environment prompt operations (MCP prompts, not task prompt)."""

    @pytest.mark.asyncio
    async def test_list_prompts_empty(self) -> None:
        """list_prompts returns empty list when no prompts."""
        from hud.environment import Environment

        env = Environment("test")

        async with env:
            prompts = await env.list_prompts()

        assert prompts == []

    @pytest.mark.asyncio
    async def test_get_prompt_not_found(self) -> None:
        """get_prompt raises when prompt not found."""
        from hud.environment import Environment

        env = Environment("test")

        async with env:
            with pytest.raises(ValueError, match="Prompt not found"):
                await env.get_prompt("nonexistent")


class TestEnvironmentSetupEvaluate:
    """Tests for setup_tool and evaluate_tool methods."""

    def test_setup_tool_with_name_and_kwargs(self) -> None:
        """setup_tool accepts name and kwargs."""
        from hud.environment import Environment

        env = Environment("test")
        env.setup_tool("navigate", url="https://example.com")

        assert len(env._setup_calls) == 1
        assert env._setup_calls[0] == ("navigate", {"url": "https://example.com"})

    def test_setup_tool_returns_self(self) -> None:
        """setup_tool returns self for chaining."""
        from hud.environment import Environment

        env = Environment("test")
        result = env.setup_tool("navigate", url="https://example.com")

        assert result is env

    def test_evaluate_tool_with_name_and_kwargs(self) -> None:
        """evaluate_tool accepts name and kwargs."""
        from hud.environment import Environment

        env = Environment("test")
        env.evaluate_tool("check_text", contains="success")

        assert len(env._evaluate_calls) == 1
        assert env._evaluate_calls[0] == ("check_text", {"contains": "success"})

    def test_evaluate_tool_returns_self(self) -> None:
        """evaluate_tool returns self for chaining."""
        from hud.environment import Environment

        env = Environment("test")
        result = env.evaluate_tool("check_text", contains="success")

        assert result is env

    def test_chaining_multiple_setup_calls(self) -> None:
        """Multiple setup_tool calls can be chained."""
        from hud.environment import Environment

        env = (
            Environment("test")
            .setup_tool("navigate", url="https://example.com")
            .setup_tool("wait", seconds=2)
        )

        assert len(env._setup_calls) == 2
