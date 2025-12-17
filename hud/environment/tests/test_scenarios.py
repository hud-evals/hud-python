"""Tests for Environment scenario decorator."""

from __future__ import annotations

import pytest

from hud.environment import Environment


class TestScenarioDecorator:
    """Tests for @env.scenario decorator."""

    def test_scenario_registers_function(self) -> None:
        """@env.scenario registers the function."""
        env = Environment("test-env")

        @env.scenario("greet")
        async def greet_scenario(name: str):
            yield f"Hello, {name}!"
            yield 1.0

        assert "greet" in env._scenarios

    def test_scenario_creates_mcp_prompt(self) -> None:
        """@env.scenario creates an MCP prompt."""
        env = Environment("test-env")

        @env.scenario("greet", description="Greeting scenario")
        async def greet_scenario(name: str):
            yield f"Hello, {name}!"
            yield 1.0

        # Check that prompt was registered via prompt manager
        prompt_names = list(env._prompt_manager._prompts.keys())
        assert "test-env:greet" in prompt_names

    def test_scenario_creates_mcp_resource(self) -> None:
        """@env.scenario creates an MCP resource."""
        env = Environment("test-env")

        @env.scenario("greet")
        async def greet_scenario(name: str):
            yield f"Hello, {name}!"
            yield 1.0

        # Check that resource was registered via resource manager
        resource_uris = list(env._resource_manager._resources.keys())
        assert "test-env:greet" in resource_uris

    def test_scenario_extracts_arguments(self) -> None:
        """@env.scenario extracts function arguments for prompt."""
        env = Environment("test-env")

        @env.scenario("checkout")
        async def checkout_scenario(user_id: str, amount: int = 100):
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


class TestScenarioExecution:
    """Tests for scenario execution flow."""

    @pytest.mark.asyncio
    async def test_scenario_setup_phase(self) -> None:
        """Scenario setup phase yields prompt."""
        env = Environment("test-env")
        setup_ran = False

        @env.scenario("test")
        async def test_scenario():
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
    async def test_scenario_stores_session(self) -> None:
        """Scenario stores generator in session for evaluate phase."""
        env = Environment("test-env")

        @env.scenario("test")
        async def test_scenario():
            yield "Test prompt"
            yield 1.0

        # Run setup via prompt - no need for context
        prompt = env._prompt_manager._prompts.get("test-env:test")
        assert prompt is not None
        await prompt.render({})

        # Check session was stored
        assert "test" in env._scenario_latest

    @pytest.mark.asyncio
    async def test_scenario_full_flow(self) -> None:
        """Scenario runs setup and evaluate phases correctly."""
        env = Environment("test-env")
        phases = []

        @env.scenario("test")
        async def test_scenario():
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


class TestScenarioWithArgs:
    """Tests for scenarios with arguments."""

    @pytest.mark.asyncio
    async def test_scenario_receives_args(self) -> None:
        """Scenario receives arguments from prompt call."""
        env = Environment("test-env")
        received_args = {}

        @env.scenario("checkout")
        async def checkout_scenario(user_id: str, amount: int = 100):
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


class TestScenarioSubmit:
    """Tests for scenario submit and answer flow."""

    @pytest.mark.asyncio
    async def test_submit_stores_answer(self) -> None:
        """submit() stores answer for scenario."""
        env = Environment("test-env")

        @env.scenario("test")
        async def test_scenario():
            yield "What is 2+2?"
            yield 1.0

        # Run setup
        prompt = env._prompt_manager._prompts.get("test-env:test")
        assert prompt is not None
        await prompt.render({})

        # Submit answer
        await env.submit("test", "4")

        assert env._scenario_answers.get("test") == "4"

    @pytest.mark.asyncio
    async def test_scenario_receives_answer(self) -> None:
        """Scenario receives submitted answer via yield."""
        env = Environment("test-env")
        received_answer = None

        @env.scenario("qa")
        async def qa_scenario():
            nonlocal received_answer
            answer = yield "What is 2+2?"
            received_answer = answer
            yield 1.0 if answer == "4" else 0.0

        # Run setup
        prompt = env._prompt_manager._prompts.get("test-env:qa")
        assert prompt is not None
        await prompt.render({})

        # Submit answer
        env._scenario_answers["qa"] = "4"

        # Run evaluate
        resource = env._resource_manager._resources.get("test-env:qa")
        assert resource is not None
        await resource.read()

        assert received_answer == "4"

    @pytest.mark.asyncio
    async def test_scenario_evaluates_answer(self) -> None:
        """Scenario evaluates answer and returns reward."""
        env = Environment("test-env")

        @env.scenario("grading")
        async def grading_scenario():
            answer = yield "What is the capital of France?"
            yield 1.0 if "paris" in answer.lower() else 0.0

        # Run setup
        prompt = env._prompt_manager._prompts.get("test-env:grading")
        assert prompt is not None
        await prompt.render({})

        # Submit correct answer
        env._scenario_answers["grading"] = "Paris"

        # Run evaluate
        resource = env._resource_manager._resources.get("test-env:grading")
        assert resource is not None
        result = await resource.read()

        import json

        data = json.loads(result)
        assert data["reward"] == 1.0


class TestScenarioMeta:
    """Tests for scenario _meta containing code."""

    def test_scenario_captures_source_code(self) -> None:
        """@env.scenario captures function source in meta."""
        env = Environment("test-env")

        @env.scenario("example")
        async def example_scenario(x: int):
            yield f"Process {x}"
            yield 1.0

        prompt = env._prompt_manager._prompts.get("test-env:example")
        assert prompt is not None
        assert prompt.meta is not None
        assert "code" in prompt.meta
        assert "async def example_scenario" in prompt.meta["code"]
        assert "yield" in prompt.meta["code"]

    def test_scenario_meta_on_resource(self) -> None:
        """Resource also has source code in meta."""
        env = Environment("test-env")

        @env.scenario("example")
        async def example_scenario():
            yield "Test"
            yield 1.0

        resource = env._resource_manager._resources.get("test-env:example")
        assert resource is not None
        assert resource.meta is not None
        assert "code" in resource.meta
        assert "async def example_scenario" in resource.meta["code"]
