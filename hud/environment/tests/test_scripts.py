"""Tests for Environment script decorator."""

from __future__ import annotations

import pytest

from hud.environment import Environment


class TestScriptDecorator:
    """Tests for @env.script decorator."""

    def test_script_registers_function(self) -> None:
        """@env.script registers the function."""
        env = Environment("test-env")

        @env.script("greet")
        async def greet_script(name: str):
            yield f"Hello, {name}!"
            yield 1.0

        assert "greet" in env._scripts

    def test_script_creates_mcp_prompt(self) -> None:
        """@env.script creates an MCP prompt."""
        env = Environment("test-env")

        @env.script("greet", description="Greeting script")
        async def greet_script(name: str):
            yield f"Hello, {name}!"
            yield 1.0

        # Check that prompt was registered via prompt manager
        prompt_names = list(env._prompt_manager._prompts.keys())
        assert "test-env:greet" in prompt_names

    def test_script_creates_mcp_resource(self) -> None:
        """@env.script creates an MCP resource."""
        env = Environment("test-env")

        @env.script("greet")
        async def greet_script(name: str):
            yield f"Hello, {name}!"
            yield 1.0

        # Check that resource was registered via resource manager
        resource_uris = list(env._resource_manager._resources.keys())
        assert "test-env:greet" in resource_uris

    def test_script_extracts_arguments(self) -> None:
        """@env.script extracts function arguments for prompt."""
        env = Environment("test-env")

        @env.script("checkout")
        async def checkout_script(user_id: str, amount: int = 100):
            yield f"Checkout for {user_id}: ${amount}"
            yield 1.0

        # Find the prompt
        prompt = env._prompt_manager._prompts.get("test-env:checkout")
        assert prompt is not None

        # Check arguments
        arg_names = [arg.name for arg in prompt.arguments]
        assert "user_id" in arg_names
        assert "amount" in arg_names


class TestScriptExecution:
    """Tests for script execution flow."""

    @pytest.mark.asyncio
    async def test_script_setup_phase(self) -> None:
        """Script setup phase yields prompt."""
        env = Environment("test-env")
        setup_ran = False

        @env.script("test")
        async def test_script():
            nonlocal setup_ran
            setup_ran = True
            yield "Test prompt"
            yield 1.0

        # Get the prompt handler
        prompt = env._prompt_manager._prompts.get("test-env:test")
        assert prompt is not None

        # Run setup via prompt render (which calls fn) - no need for context
        result = await prompt.render({})

        assert setup_ran
        # Result is list of PromptMessage
        assert len(result) > 0
        assert "Test prompt" in str(result[0].content)

    @pytest.mark.asyncio
    async def test_script_stores_session(self) -> None:
        """Script stores generator in session for evaluate phase."""
        env = Environment("test-env")

        @env.script("test")
        async def test_script():
            yield "Test prompt"
            yield 1.0

        # Run setup via prompt - no need for context
        prompt = env._prompt_manager._prompts.get("test-env:test")
        await prompt.render({})

        # Check session was stored
        assert "test" in env._script_latest

    @pytest.mark.asyncio
    async def test_script_full_flow(self) -> None:
        """Script runs setup and evaluate phases correctly."""
        env = Environment("test-env")
        phases = []

        @env.script("test")
        async def test_script():
            phases.append("setup")
            yield "Test prompt"
            phases.append("evaluate")
            yield 0.95

        # Setup phase - no context needed for prompt/resource
        prompt = env._prompt_manager._prompts.get("test-env:test")
        await prompt.render({})
        assert "setup" in phases
        assert "evaluate" not in phases

        # Evaluate phase
        resource = env._resource_manager._resources.get("test-env:test")
        reward_result = await resource.read()
        assert "evaluate" in phases


class TestScriptWithArgs:
    """Tests for scripts with arguments."""

    @pytest.mark.asyncio
    async def test_script_receives_args(self) -> None:
        """Script receives arguments from prompt call."""
        env = Environment("test-env")
        received_args = {}

        @env.script("checkout")
        async def checkout_script(user_id: str, amount: int = 100):
            received_args["user_id"] = user_id
            received_args["amount"] = amount
            yield f"Checkout {user_id}: ${amount}"
            yield 1.0

        prompt = env._prompt_manager._prompts.get("test-env:checkout")

        # No context needed for prompt render
        await prompt.render({"user_id": "alice", "amount": 50})

        assert received_args["user_id"] == "alice"
        assert received_args["amount"] == 50

