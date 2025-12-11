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
        assert prompt.arguments is not None

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
        assert prompt is not None
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
        assert prompt is not None
        await prompt.render({})
        assert "setup" in phases
        assert "evaluate" not in phases

        # Evaluate phase
        resource = env._resource_manager._resources.get("test-env:test")
        assert resource is not None
        await resource.read()
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
        assert prompt is not None
        # No context needed for prompt render
        await prompt.render({"user_id": "alice", "amount": 50})

        assert received_args["user_id"] == "alice"
        assert received_args["amount"] == 50


class TestScriptSubmit:
    """Tests for script submit and answer flow."""

    @pytest.mark.asyncio
    async def test_submit_stores_answer(self) -> None:
        """submit() stores answer for script."""
        env = Environment("test-env")

        @env.script("test")
        async def test_script():
            yield "What is 2+2?"
            yield 1.0

        # Run setup
        prompt = env._prompt_manager._prompts.get("test-env:test")
        assert prompt is not None
        await prompt.render({})

        # Submit answer
        await env.submit("test", "4")

        assert env._script_answers.get("test") == "4"

    @pytest.mark.asyncio
    async def test_script_receives_answer(self) -> None:
        """Script receives submitted answer via yield."""
        env = Environment("test-env")
        received_answer = None

        @env.script("qa")
        async def qa_script():
            nonlocal received_answer
            answer = yield "What is 2+2?"
            received_answer = answer
            yield 1.0 if answer == "4" else 0.0

        # Run setup
        prompt = env._prompt_manager._prompts.get("test-env:qa")
        assert prompt is not None
        await prompt.render({})

        # Submit answer
        env._script_answers["qa"] = "4"

        # Run evaluate
        resource = env._resource_manager._resources.get("test-env:qa")
        assert resource is not None
        await resource.read()

        assert received_answer == "4"

    @pytest.mark.asyncio
    async def test_script_evaluates_answer(self) -> None:
        """Script evaluates answer and returns reward."""
        env = Environment("test-env")

        @env.script("grading")
        async def grading_script():
            answer = yield "What is the capital of France?"
            yield 1.0 if "paris" in answer.lower() else 0.0

        # Run setup
        prompt = env._prompt_manager._prompts.get("test-env:grading")
        assert prompt is not None
        await prompt.render({})

        # Submit correct answer
        env._script_answers["grading"] = "Paris"

        # Run evaluate
        resource = env._resource_manager._resources.get("test-env:grading")
        assert resource is not None
        result = await resource.read()

        import json

        data = json.loads(result)
        assert data["reward"] == 1.0


class TestScriptMeta:
    """Tests for script _meta containing code."""

    def test_script_captures_source_code(self) -> None:
        """@env.script captures function source in meta."""
        env = Environment("test-env")

        @env.script("example")
        async def example_script(x: int):
            yield f"Process {x}"
            yield 1.0

        prompt = env._prompt_manager._prompts.get("test-env:example")
        assert prompt is not None
        assert prompt.meta is not None
        assert "code" in prompt.meta
        assert "async def example_script" in prompt.meta["code"]
        assert "yield" in prompt.meta["code"]

    def test_script_meta_on_resource(self) -> None:
        """Resource also has source code in meta."""
        env = Environment("test-env")

        @env.script("example")
        async def example_script():
            yield "Test"
            yield 1.0

        resource = env._resource_manager._resources.get("test-env:example")
        assert resource is not None
        assert resource.meta is not None
        assert "code" in resource.meta
        assert "async def example_script" in resource.meta["code"]
