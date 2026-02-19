"""Tests for Environment scenario decorator."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

import pytest
from pydantic import BaseModel

from hud.environment import Environment


# Module-level models for Pydantic/Enum/datetime deserialization tests
# (prefixed with underscore to avoid pytest collection warnings)
class _UserConfig(BaseModel):
    """Pydantic model for testing."""

    name: str
    age: int
    active: bool = True


class _Status(Enum):
    """Enum for testing."""

    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"


class _Address(BaseModel):
    """Nested Pydantic model for testing."""

    street: str
    city: str


class _Person(BaseModel):
    """Pydantic model with nested model for testing."""

    name: str
    address: _Address


class _Item(BaseModel):
    """Pydantic model for list tests."""

    id: int
    name: str


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

        # Check session was stored in _active_session
        assert env._active_session is not None
        assert env._active_session.local_name == "test"

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
        """submit() stores answer in active session."""
        env = Environment("test-env")

        @env.scenario("test")
        async def test_scenario():
            yield "What is 2+2?"
            yield 1.0

        # Run setup via proper API (creates _active_session)
        await env.run_scenario_setup("test", {})

        # Submit answer
        await env.submit("test", "4")

        # Answer is stored in active session (not _scenario_answers for client-side)
        assert env._active_session is not None
        assert env._active_session.answer == "4"

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

        # Run setup (creates _active_session)
        prompt = env._prompt_manager._prompts.get("test-env:qa")
        assert prompt is not None
        await prompt.render({})

        # Submit answer via _active_session
        assert env._active_session is not None
        env._active_session.answer = "4"

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

        # Run setup (creates _active_session)
        prompt = env._prompt_manager._prompts.get("test-env:grading")
        assert prompt is not None
        await prompt.render({})

        # Submit correct answer via _active_session
        assert env._active_session is not None
        env._active_session.answer = "Paris"

        # Run evaluate
        resource = env._resource_manager._resources.get("test-env:grading")
        assert resource is not None
        result = await resource.read()

        import json

        data = json.loads(result)
        assert data["reward"] == 1.0

    @pytest.mark.asyncio
    async def test_hud_submit_normalizes_prefixed_scenario_name(self) -> None:
        """_hud_submit with prefixed name stores answer in _active_session.

        Regression test: answers submitted with "env:scenario" prefix must
        match the active session's local_name for storage.
        """
        env = Environment("my-env")

        @env.scenario("greet")
        async def greet_scenario():
            answer = yield "Say hello"
            yield 1.0 if answer == "hello" else 0.0

        # Run setup via prompt (creates _active_session)
        prompt = env._prompt_manager._prompts.get("my-env:greet")
        assert prompt is not None
        await prompt.render({})

        # Verify session exists before _hud_submit
        assert env._active_session is not None
        assert env._active_session.local_name == "greet"

        # Simulate _hud_submit with PREFIXED scenario name (as happens in remote calls)
        # This should normalize to "greet" and match the active session
        await env.call_tool("_hud_submit", scenario="my-env:greet", answer="hello")

        # Verify answer was stored in _active_session
        assert env._active_session.answer == "hello"

        # Verify evaluation works
        resource = env._resource_manager._resources.get("my-env:greet")
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


class TestScenarioJsonSerialization:
    """Tests for JSON serialization of complex argument types.

    MCP prompts only support string arguments (dict[str, str]).
    Complex types like lists, dicts, and numbers are JSON-serialized
    when sent and deserialized based on type annotations when received.
    """

    @pytest.mark.asyncio
    async def test_list_argument_deserialization(self) -> None:
        """List arguments are JSON-deserialized from strings."""
        env = Environment("test-env")
        received_items: list[str] = []

        @env.scenario("process_items")
        async def process_items_scenario(items: list[str]):
            received_items.extend(items)
            yield f"Processing {len(items)} items"
            yield 1.0

        prompt = env._prompt_manager._prompts.get("test-env:process_items")
        assert prompt is not None

        # Simulate MCP sending JSON-encoded list as string
        await prompt.render({"items": '["apple", "banana", "cherry"]'})

        assert received_items == ["apple", "banana", "cherry"]

    @pytest.mark.asyncio
    async def test_dict_argument_deserialization(self) -> None:
        """Dict arguments are JSON-deserialized from strings."""
        env = Environment("test-env")
        received_config: dict[str, Any] = {}

        @env.scenario("configure")
        async def configure_scenario(config: dict[str, Any]):
            received_config.update(config)
            yield "Configuring..."
            yield 1.0

        prompt = env._prompt_manager._prompts.get("test-env:configure")
        assert prompt is not None

        # Simulate MCP sending JSON-encoded dict as string
        await prompt.render({"config": '{"timeout": 30, "retries": 3}'})

        assert received_config == {"timeout": 30, "retries": 3}

    @pytest.mark.asyncio
    async def test_int_argument_deserialization(self) -> None:
        """Integer arguments are JSON-deserialized from strings."""
        env = Environment("test-env")
        received_count = 0

        @env.scenario("count")
        async def count_scenario(count: int):
            nonlocal received_count
            received_count = count
            yield f"Counting to {count}"
            yield 1.0

        prompt = env._prompt_manager._prompts.get("test-env:count")
        assert prompt is not None

        # Simulate MCP sending JSON-encoded int as string
        await prompt.render({"count": "42"})

        assert received_count == 42
        assert isinstance(received_count, int)

    @pytest.mark.asyncio
    async def test_float_argument_deserialization(self) -> None:
        """Float arguments are JSON-deserialized from strings."""
        env = Environment("test-env")
        received_value = 0.0

        @env.scenario("precision")
        async def precision_scenario(value: float):
            nonlocal received_value
            received_value = value
            yield f"Value is {value}"
            yield 1.0

        prompt = env._prompt_manager._prompts.get("test-env:precision")
        assert prompt is not None

        # Simulate MCP sending JSON-encoded float as string
        await prompt.render({"value": "3.14159"})

        assert received_value == 3.14159
        assert isinstance(received_value, float)

    @pytest.mark.asyncio
    async def test_bool_argument_deserialization(self) -> None:
        """Boolean arguments are JSON-deserialized from strings."""
        env = Environment("test-env")
        received_flag = False

        @env.scenario("toggle")
        async def toggle_scenario(enabled: bool):
            nonlocal received_flag
            received_flag = enabled
            yield f"Enabled: {enabled}"
            yield 1.0

        prompt = env._prompt_manager._prompts.get("test-env:toggle")
        assert prompt is not None

        # Simulate MCP sending JSON-encoded bool as string
        await prompt.render({"enabled": "true"})

        assert received_flag is True
        assert isinstance(received_flag, bool)

    @pytest.mark.asyncio
    async def test_string_argument_unchanged(self) -> None:
        """String arguments are passed through unchanged."""
        env = Environment("test-env")
        received_name = ""

        @env.scenario("greet")
        async def greet_scenario(name: str):
            nonlocal received_name
            received_name = name
            yield f"Hello, {name}!"
            yield 1.0

        prompt = env._prompt_manager._prompts.get("test-env:greet")
        assert prompt is not None

        # String should pass through as-is (not double-encoded)
        await prompt.render({"name": "Alice"})

        assert received_name == "Alice"

    @pytest.mark.asyncio
    async def test_mixed_argument_types(self) -> None:
        """Mixed argument types are handled correctly."""
        env = Environment("test-env")
        received_args: dict[str, Any] = {}

        @env.scenario("mixed")
        async def mixed_scenario(
            name: str,
            count: int,
            items: list[str],
            options: dict[str, bool],
        ):
            received_args["name"] = name
            received_args["count"] = count
            received_args["items"] = items
            received_args["options"] = options
            yield "Processing..."
            yield 1.0

        prompt = env._prompt_manager._prompts.get("test-env:mixed")
        assert prompt is not None

        await prompt.render(
            {
                "name": "test",
                "count": "5",
                "items": '["a", "b", "c"]',
                "options": '{"verbose": true, "dry_run": false}',
            }
        )

        assert received_args["name"] == "test"
        assert received_args["count"] == 5
        assert received_args["items"] == ["a", "b", "c"]
        assert received_args["options"] == {"verbose": True, "dry_run": False}

    @pytest.mark.asyncio
    async def test_invalid_json_falls_back_to_string(self) -> None:
        """Invalid JSON for non-string type falls back to string value."""
        env = Environment("test-env")
        received_items: list[str] = []

        @env.scenario("fallback")
        async def fallback_scenario(items: list[str]):
            # This will receive the raw string if JSON parsing fails
            received_items.append(str(items))
            yield "Processing..."
            yield 1.0

        prompt = env._prompt_manager._prompts.get("test-env:fallback")
        assert prompt is not None

        # Invalid JSON - should fall back to string
        await prompt.render({"items": "not valid json ["})

        # Falls back to raw string
        assert received_items == ["not valid json ["]

    @pytest.mark.asyncio
    async def test_nested_complex_types(self) -> None:
        """Nested complex types are deserialized correctly."""
        env = Environment("test-env")
        received_data: dict[str, Any] = {}

        @env.scenario("nested")
        async def nested_scenario(data: dict[str, Any]):
            received_data.update(data)
            yield "Processing nested data..."
            yield 1.0

        prompt = env._prompt_manager._prompts.get("test-env:nested")
        assert prompt is not None

        nested_json = (
            '{"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}], '
            '"metadata": {"version": 1}}'
        )
        await prompt.render({"data": nested_json})

        assert received_data == {
            "users": [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25},
            ],
            "metadata": {"version": 1},
        }

    @pytest.mark.asyncio
    async def test_optional_list_with_value(self) -> None:
        """Optional[list[str]] receives list when provided."""
        env = Environment("test-env")
        received_items: list[str] | None = None

        @env.scenario("optional_list")
        async def optional_list_scenario(items: list[str] | None = None):
            nonlocal received_items
            received_items = items
            yield f"Got {items}"
            yield 1.0

        prompt = env._prompt_manager._prompts.get("test-env:optional_list")
        assert prompt is not None

        await prompt.render({"items": '["x", "y", "z"]'})

        assert received_items == ["x", "y", "z"]

    @pytest.mark.asyncio
    async def test_optional_list_with_null(self) -> None:
        """Optional[list[str]] receives None when 'null' is passed."""
        env = Environment("test-env")
        received_items: list[str] | None = ["initial"]

        @env.scenario("optional_list_null")
        async def optional_list_null_scenario(items: list[str] | None = None):
            nonlocal received_items
            received_items = items
            yield f"Got {items}"
            yield 1.0

        prompt = env._prompt_manager._prompts.get("test-env:optional_list_null")
        assert prompt is not None

        await prompt.render({"items": "null"})

        assert received_items is None

    @pytest.mark.asyncio
    async def test_optional_str_with_value(self) -> None:
        """Optional[str] receives string value correctly."""
        env = Environment("test-env")
        received_name: str | None = None

        @env.scenario("optional_str")
        async def optional_str_scenario(name: str | None = None):
            nonlocal received_name
            received_name = name
            yield f"Got {name}"
            yield 1.0

        prompt = env._prompt_manager._prompts.get("test-env:optional_str")
        assert prompt is not None

        await prompt.render({"name": "Alice"})

        assert received_name == "Alice"

    @pytest.mark.asyncio
    async def test_optional_str_with_null(self) -> None:
        """Optional[str] receives None when 'null' is passed."""
        env = Environment("test-env")
        received_name: str | None = "initial"

        @env.scenario("optional_str_null")
        async def optional_str_null_scenario(name: str | None = None):
            nonlocal received_name
            received_name = name
            yield f"Got {name}"
            yield 1.0

        prompt = env._prompt_manager._prompts.get("test-env:optional_str_null")
        assert prompt is not None

        await prompt.render({"name": "null"})

        assert received_name is None

    @pytest.mark.asyncio
    async def test_pydantic_model_deserialization(self) -> None:
        """Pydantic models are properly deserialized from JSON."""
        env = Environment("test-env")
        received_config: _UserConfig | None = None

        @env.scenario("pydantic_model")
        async def pydantic_model_scenario(config: _UserConfig):
            nonlocal received_config
            received_config = config
            yield f"Got config for {config.name}"
            yield 1.0

        prompt = env._prompt_manager._prompts.get("test-env:pydantic_model")
        assert prompt is not None

        await prompt.render({"config": '{"name": "Alice", "age": 30}'})

        assert received_config is not None
        assert isinstance(received_config, _UserConfig)
        assert received_config.name == "Alice"
        assert received_config.age == 30
        assert received_config.active is True  # default value

    @pytest.mark.asyncio
    async def test_enum_deserialization(self) -> None:
        """Enum values are properly deserialized from JSON strings."""
        env = Environment("test-env")
        received_status: _Status | None = None

        @env.scenario("enum_status")
        async def enum_scenario(status: _Status):
            nonlocal received_status
            received_status = status
            yield f"Status is {status.value}"
            yield 1.0

        prompt = env._prompt_manager._prompts.get("test-env:enum_status")
        assert prompt is not None

        await prompt.render({"status": '"active"'})

        assert received_status is not None
        assert isinstance(received_status, _Status)
        assert received_status == _Status.ACTIVE

    @pytest.mark.asyncio
    async def test_datetime_deserialization(self) -> None:
        """Datetime values are properly deserialized from ISO strings."""
        env = Environment("test-env")
        received_dt: datetime | None = None

        @env.scenario("datetime_scenario")
        async def datetime_scenario(created_at: datetime):
            nonlocal received_dt
            received_dt = created_at
            yield f"Created at {created_at}"
            yield 1.0

        prompt = env._prompt_manager._prompts.get("test-env:datetime_scenario")
        assert prompt is not None

        await prompt.render({"created_at": '"2024-06-15T10:30:00"'})

        assert received_dt is not None
        assert isinstance(received_dt, datetime)
        assert received_dt.year == 2024
        assert received_dt.month == 6
        assert received_dt.day == 15
        assert received_dt.hour == 10
        assert received_dt.minute == 30

    @pytest.mark.asyncio
    async def test_nested_pydantic_model(self) -> None:
        """Nested Pydantic models are properly deserialized."""
        env = Environment("test-env")
        received_person: _Person | None = None

        @env.scenario("nested_pydantic")
        async def nested_pydantic_scenario(person: _Person):
            nonlocal received_person
            received_person = person
            yield f"Person {person.name} from {person.address.city}"
            yield 1.0

        prompt = env._prompt_manager._prompts.get("test-env:nested_pydantic")
        assert prompt is not None

        json_data = '{"name": "Bob", "address": {"street": "123 Main St", "city": "NYC"}}'
        await prompt.render({"person": json_data})

        assert received_person is not None
        assert isinstance(received_person, _Person)
        assert received_person.name == "Bob"
        assert isinstance(received_person.address, _Address)
        assert received_person.address.city == "NYC"

    @pytest.mark.asyncio
    async def test_list_of_pydantic_models(self) -> None:
        """List of Pydantic models are properly deserialized."""
        env = Environment("test-env")
        received_items: list[_Item] = []

        @env.scenario("list_pydantic")
        async def list_pydantic_scenario(items: list[_Item]):
            nonlocal received_items
            received_items = items
            yield f"Got {len(items)} items"
            yield 1.0

        prompt = env._prompt_manager._prompts.get("test-env:list_pydantic")
        assert prompt is not None

        json_data = '[{"id": 1, "name": "Apple"}, {"id": 2, "name": "Banana"}]'
        await prompt.render({"items": json_data})

        assert len(received_items) == 2
        assert all(isinstance(item, _Item) for item in received_items)
        assert received_items[0].name == "Apple"
        assert received_items[1].name == "Banana"


class TestLiteralDeserialization:
    """Tests for Literal type deserialization edge cases.

    The MCP protocol sends all arguments as strings. When the scenario
    function uses Literal types, the deserializer must correctly match
    string values -- especially numeric-looking strings like "0", "1".
    """

    @pytest.mark.asyncio
    async def test_literal_string_kept_as_string(self) -> None:
        """Literal["a", "b"] receives string values correctly."""
        env = Environment("test-env")
        received: str | None = None

        @env.scenario("literal_str")
        async def literal_str_scenario(choice: Literal["a", "b"]):
            nonlocal received
            received = choice
            yield f"Got {choice}"
            yield 1.0

        prompt = env._prompt_manager._prompts.get("test-env:literal_str")
        assert prompt is not None

        await prompt.render({"choice": "a"})
        assert received == "a"
        assert isinstance(received, str)

    @pytest.mark.asyncio
    async def test_literal_numeric_string_not_coerced_to_int(self) -> None:
        """Literal["0", "1", "2"] keeps "0" as string, not int 0.

        This is the GPQA Diamond bug: task IDs are "0", "1", etc.
        and must stay as strings for Path operations.
        """
        env = Environment("test-env")
        received: Any = None

        @env.scenario("literal_numeric")
        async def literal_numeric_scenario(task_id: Literal["0", "1", "2"]):
            nonlocal received
            received = task_id
            yield f"Task {task_id}"
            yield 1.0

        prompt = env._prompt_manager._prompts.get("test-env:literal_numeric")
        assert prompt is not None

        await prompt.render({"task_id": "0"})
        assert received == "0"
        assert isinstance(received, str)

    @pytest.mark.asyncio
    async def test_literal_numeric_string_various_values(self) -> None:
        """All numeric-looking Literal string values stay as strings."""
        env = Environment("test-env")
        received: Any = None

        @env.scenario("literal_nums")
        async def literal_nums_scenario(idx: Literal["0", "42", "197"]):
            nonlocal received
            received = idx
            yield f"Index {idx}"
            yield 1.0

        prompt = env._prompt_manager._prompts.get("test-env:literal_nums")
        assert prompt is not None

        for val in ("0", "42", "197"):
            await prompt.render({"idx": val})
            assert received == val, f"Expected {val!r}, got {received!r}"
            assert isinstance(received, str), f"Expected str, got {type(received)}"

    @pytest.mark.asyncio
    async def test_literal_int_coerces_correctly(self) -> None:
        """Literal[1, 2, 3] with int values coerces string "1" to int 1."""
        env = Environment("test-env")
        received: Any = None

        @env.scenario("literal_int")
        async def literal_int_scenario(level: Literal[1, 2, 3]):
            nonlocal received
            received = level
            yield f"Level {level}"
            yield 1.0

        prompt = env._prompt_manager._prompts.get("test-env:literal_int")
        assert prompt is not None

        await prompt.render({"level": "2"})
        assert received == 2
        assert isinstance(received, int)

    @pytest.mark.asyncio
    async def test_literal_mixed_types(self) -> None:
        """Literal["auto", 0, 1] handles mixed string/int literal values."""
        env = Environment("test-env")
        received: Any = None

        @env.scenario("literal_mixed")
        async def literal_mixed_scenario(mode: Literal["auto", 0, 1]):
            nonlocal received
            received = mode
            yield f"Mode {mode}"
            yield 1.0

        prompt = env._prompt_manager._prompts.get("test-env:literal_mixed")
        assert prompt is not None

        await prompt.render({"mode": "auto"})
        assert received == "auto"

    @pytest.mark.asyncio
    async def test_literal_with_default(self) -> None:
        """Literal with default value works when arg is provided."""
        env = Environment("test-env")
        received: Any = None

        @env.scenario("literal_default")
        async def literal_default_scenario(
            task_id: Literal["build-pmars"] = "build-pmars",
        ):
            nonlocal received
            received = task_id
            yield f"Task {task_id}"
            yield 1.0

        prompt = env._prompt_manager._prompts.get("test-env:literal_default")
        assert prompt is not None

        await prompt.render({"task_id": "build-pmars"})
        assert received == "build-pmars"

    @pytest.mark.asyncio
    async def test_int_annotation_coerces_numeric_string(self) -> None:
        """Plain int annotation coerces "42" to 42."""
        env = Environment("test-env")
        received: Any = None

        @env.scenario("int_arg")
        async def int_arg_scenario(count: int):
            nonlocal received
            received = count
            yield f"Count {count}"
            yield 1.0

        prompt = env._prompt_manager._prompts.get("test-env:int_arg")
        assert prompt is not None

        await prompt.render({"count": "42"})
        assert received == 42
        assert isinstance(received, int)

    @pytest.mark.asyncio
    async def test_float_annotation_coerces_numeric_string(self) -> None:
        """Plain float annotation coerces "3.14" to 3.14."""
        env = Environment("test-env")
        received: Any = None

        @env.scenario("float_arg")
        async def float_arg_scenario(rate: float):
            nonlocal received
            received = rate
            yield f"Rate {rate}"
            yield 1.0

        prompt = env._prompt_manager._prompts.get("test-env:float_arg")
        assert prompt is not None

        await prompt.render({"rate": "3.14"})
        assert received == pytest.approx(3.14)
        assert isinstance(received, float)

    @pytest.mark.asyncio
    async def test_bool_annotation_coerces_string(self) -> None:
        """Bool annotation coerces "true"/"false" correctly."""
        env = Environment("test-env")
        received: Any = None

        @env.scenario("bool_arg")
        async def bool_arg_scenario(verbose: bool):
            nonlocal received
            received = verbose
            yield f"Verbose {verbose}"
            yield 1.0

        prompt = env._prompt_manager._prompts.get("test-env:bool_arg")
        assert prompt is not None

        await prompt.render({"verbose": "true"})
        assert received is True

    @pytest.mark.asyncio
    async def test_str_annotation_preserves_numeric_string(self) -> None:
        """Plain str annotation keeps "42" as string "42"."""
        env = Environment("test-env")
        received: Any = None

        @env.scenario("str_numeric")
        async def str_numeric_scenario(name: str):
            nonlocal received
            received = name
            yield f"Name {name}"
            yield 1.0

        prompt = env._prompt_manager._prompts.get("test-env:str_numeric")
        assert prompt is not None

        await prompt.render({"name": "42"})
        assert received == "42"
        assert isinstance(received, str)

    @pytest.mark.asyncio
    async def test_no_annotation_numeric_becomes_int(self) -> None:
        """Untyped arg with numeric-looking string falls through to json.loads."""
        env = Environment("test-env")
        received: Any = None

        @env.scenario("untyped_num")
        async def untyped_num_scenario(val):
            nonlocal received
            received = val
            yield f"Val {val}"
            yield 1.0

        prompt = env._prompt_manager._prompts.get("test-env:untyped_num")
        assert prompt is not None

        await prompt.render({"val": "42"})
        # Without annotation, generic heuristic converts to int
        assert received == 42


class TestScenarioNameNormalization:
    """Test edge cases for environment and scenario name handling."""

    @pytest.mark.asyncio
    async def test_env_name_with_underscores_normalizes(self) -> None:
        """Environment name with underscores normalizes to hyphens."""
        env = Environment("my_test_env")
        assert env.name == "my-test-env"

        @env.scenario("greet")
        async def greet():
            yield "Hello"
            yield 1.0

        # Scenario should be registered with normalized name
        assert "my-test-env:greet" in [p.name for p in env._prompt_manager._prompts.values()]

    @pytest.mark.asyncio
    async def test_env_name_with_spaces_normalizes(self) -> None:
        """Environment name with spaces normalizes to hyphens."""
        env = Environment("my test env")
        assert env.name == "my-test-env"

    @pytest.mark.asyncio
    async def test_env_name_with_caps_normalizes(self) -> None:
        """Environment name with capitals normalizes to lowercase."""
        env = Environment("MyTestEnv")
        assert env.name == "mytestenv"

    @pytest.mark.asyncio
    async def test_env_name_mixed_formatting(self) -> None:
        """Environment name with mixed formatting normalizes correctly."""
        env = Environment("My_Test Env")
        assert env.name == "my-test-env"

    @pytest.mark.asyncio
    async def test_prefix_matches_normalized_name(self) -> None:
        """Scenario prefix should match normalized env name."""
        env = Environment("my_env")  # Normalizes to "my-env"

        @env.scenario("test")
        async def test_scenario():
            yield "Prompt"
            yield 1.0

        # Calling with normalized prefix should work as local
        prompt = await env.run_scenario_setup("my-env:test", {})
        assert prompt == "Prompt"
        assert env._active_session is not None
        assert env._active_session.is_local is True

    @pytest.mark.asyncio
    async def test_unnormalized_prefix_treated_as_remote(self) -> None:
        """Calling with unnormalized prefix treats as remote (different env)."""
        env = Environment("my_env")  # Normalizes to "my-env"

        @env.scenario("test")
        async def test_scenario():
            yield "Prompt"
            yield 1.0

        # Calling with "my_env:test" (underscore) won't match "my-env"
        # So it's treated as remote - which will fail since no connection
        with pytest.raises(ValueError, match="Scenario not found"):
            await env.run_scenario_setup("my_env:test", {})


class TestScenarioMalformedNames:
    """Test handling of malformed scenario names."""

    @pytest.mark.asyncio
    async def test_empty_scenario_name_rejected(self) -> None:
        """Empty scenario name should be handled gracefully."""
        env = Environment("test-env")

        @env.scenario("valid")
        async def valid_scenario():
            yield "Prompt"
            yield 1.0

        # Empty name - should fail since not registered
        with pytest.raises((ValueError, KeyError)):
            await env.run_scenario_setup("", {})

    @pytest.mark.asyncio
    async def test_only_colon_handled(self) -> None:
        """Scenario name that is just ':' should be handled."""
        env = Environment("test-env")

        # ":" splits to prefix="" and short_name=""
        with pytest.raises((ValueError, KeyError)):
            await env.run_scenario_setup(":", {})

    @pytest.mark.asyncio
    async def test_colon_in_scenario_name_rejected_at_registration(self) -> None:
        """Scenario names with colons are rejected at registration time."""
        env = Environment("test-env")

        # Colons are reserved as the separator between env and scenario names
        with pytest.raises(ValueError, match="cannot contain ':'"):

            @env.scenario("invalid:name")
            async def scenario_with_colon():
                yield "Prompt"
                yield 1.0

    @pytest.mark.asyncio
    async def test_whitespace_in_scenario_name(self) -> None:
        """Scenario names with whitespace should work (not normalized)."""
        env = Environment("test-env")

        @env.scenario("my scenario")
        async def scenario_with_space():
            yield "Prompt"
            yield 1.0

        # Scenario names are NOT normalized (only env names are)
        prompt = await env.run_scenario_setup("my scenario", {})
        assert prompt == "Prompt"


class TestScenarioRegistration:
    """Test scenario registration edge cases."""

    @pytest.mark.asyncio
    async def test_duplicate_scenario_name_overwrites(self) -> None:
        """Registering same scenario name twice should overwrite."""
        env = Environment("test-env")

        @env.scenario("greet")
        async def greet_v1():
            yield "Hello v1"
            yield 1.0

        @env.scenario("greet")
        async def greet_v2():
            yield "Hello v2"
            yield 1.0

        # Should use v2
        prompt = await env.run_scenario_setup("greet", {})
        assert prompt == "Hello v2"

    @pytest.mark.asyncio
    async def test_scenario_with_special_chars(self) -> None:
        """Scenario names can contain special characters."""
        env = Environment("test-env")

        @env.scenario("test-scenario_v2.0")
        async def special_scenario():
            yield "Prompt"
            yield 1.0

        prompt = await env.run_scenario_setup("test-scenario_v2.0", {})
        assert prompt == "Prompt"

    @pytest.mark.asyncio
    async def test_scenario_that_yields_once(self) -> None:
        """Scenario that yields only once (no evaluate) should handle gracefully."""
        env = Environment("test-env")

        @env.scenario("one-yield")
        async def one_yield_scenario():
            yield "Prompt"
            # No second yield!

        prompt = await env.run_scenario_setup("one-yield", {})
        assert prompt == "Prompt"

        assert env._active_session is not None
        env._active_session.answer = "test"
        # Evaluate should handle StopAsyncIteration and return EvaluationResult with reward=1.0
        result = await env.run_scenario_evaluate("one-yield")
        assert result is not None
        assert result.reward == 1.0
        assert result.done is True

    @pytest.mark.asyncio
    async def test_scenario_that_yields_three_times(self) -> None:
        """Scenario that yields more than twice - third yield ignored."""
        env = Environment("test-env")

        @env.scenario("three-yields")
        async def three_yield_scenario():
            yield "Prompt"
            yield 0.5
            yield "This should be ignored"

        prompt = await env.run_scenario_setup("three-yields", {})
        assert prompt == "Prompt"

        assert env._active_session is not None
        env._active_session.answer = "test"
        result = await env.run_scenario_evaluate("three-yields")
        assert result is not None
        assert result.reward == 0.5


class TestScenarioSessionState:
    """Test session state management edge cases."""

    @pytest.mark.asyncio
    async def test_submit_before_setup_raises(self) -> None:
        """Calling submit() before run_scenario_setup() should raise."""
        env = Environment("test-env")

        @env.scenario("test")
        async def test_scenario():
            yield "Prompt"
            yield 1.0

        with pytest.raises(ValueError, match="No active scenario session"):
            await env.submit("test", "answer")

    @pytest.mark.asyncio
    async def test_evaluate_before_setup_raises(self) -> None:
        """Calling evaluate() before setup() should raise ValueError."""
        env = Environment("test-env")

        @env.scenario("test")
        async def test_scenario():
            yield "Prompt"
            yield 1.0

        with pytest.raises(ValueError, match="No active session"):
            await env.run_scenario_evaluate("test")

    @pytest.mark.asyncio
    async def test_double_evaluate_raises(self) -> None:
        """Calling evaluate() twice should raise ValueError on second call."""
        env = Environment("test-env")

        @env.scenario("test")
        async def test_scenario():
            yield "Prompt"
            yield 0.75

        await env.run_scenario_setup("test", {})
        assert env._active_session is not None
        env._active_session.answer = "answer"

        result1 = await env.run_scenario_evaluate("test")
        assert result1 is not None
        assert result1.reward == 0.75

        # Second call - session cleared, should raise
        with pytest.raises(ValueError, match="No active session"):
            await env.run_scenario_evaluate("test")

    @pytest.mark.asyncio
    async def test_submit_wrong_scenario_raises(self) -> None:
        """Submitting answer for wrong scenario should raise."""
        env = Environment("test-env")

        @env.scenario("scenario-a")
        async def scenario_a():
            yield "Prompt A"
            yield 1.0

        @env.scenario("scenario-b")
        async def scenario_b():
            yield "Prompt B"
            yield 1.0

        await env.run_scenario_setup("scenario-a", {})

        with pytest.raises(ValueError, match="Scenario mismatch"):
            await env.submit("scenario-b", "answer")

    @pytest.mark.asyncio
    async def test_second_setup_overwrites_first(self) -> None:
        """Starting a new scenario before evaluating previous one overwrites."""
        env = Environment("test-env")

        @env.scenario("first")
        async def first_scenario():
            yield "First"
            yield 1.0

        @env.scenario("second")
        async def second_scenario():
            yield "Second"
            yield 0.5

        await env.run_scenario_setup("first", {})
        assert env._active_session is not None
        assert env._active_session.local_name == "first"

        # Start second without evaluating first
        await env.run_scenario_setup("second", {})
        assert env._active_session is not None
        assert env._active_session.local_name == "second"

        env._active_session.answer = "answer"
        result = await env.run_scenario_evaluate("second")
        assert result is not None
        assert result.reward == 0.5


class TestEvaluationResultYield:
    """Test scenarios that yield EvaluationResult instead of float."""

    @pytest.mark.asyncio
    async def test_yield_evaluation_result(self) -> None:
        """Scenario can yield EvaluationResult directly."""
        from hud.tools.types import EvaluationResult

        env = Environment("test-env")

        @env.scenario("eval-result")
        async def eval_result_scenario():
            answer = yield "Do the task"
            yield EvaluationResult(
                reward=0.85,
                done=True,
                content=f"Received: {answer}",
            )

        prompt = await env.run_scenario_setup("eval-result", {})
        assert prompt == "Do the task"

        assert env._active_session is not None
        env._active_session.answer = "completed"
        result = await env.run_scenario_evaluate("eval-result")

        assert result is not None
        assert result.reward == 0.85
        assert result.done is True
        assert result.content == "Received: completed"

    @pytest.mark.asyncio
    async def test_yield_evaluation_result_with_subscores(self) -> None:
        """Scenario can yield EvaluationResult with subscores."""
        from hud.tools.types import EvaluationResult, SubScore

        env = Environment("test-env")

        @env.scenario("with-subscores")
        async def subscores_scenario():
            yield "Complete the task"
            yield EvaluationResult(
                reward=0.75,
                done=True,
                subscores=[
                    SubScore(name="accuracy", weight=0.6, value=0.8),
                    SubScore(name="speed", weight=0.4, value=0.7),
                ],
            )

        await env.run_scenario_setup("with-subscores", {})
        assert env._active_session is not None
        env._active_session.answer = "done"
        result = await env.run_scenario_evaluate("with-subscores")

        assert result is not None
        assert result.reward == 0.75
        assert result.subscores is not None
        assert len(result.subscores) == 2
        assert result.subscores[0].name == "accuracy"
        assert result.subscores[0].weight == 0.6
        assert result.subscores[0].value == 0.8

    @pytest.mark.asyncio
    async def test_yield_evaluation_result_partial_done(self) -> None:
        """Scenario can indicate partial completion with done=False."""
        from hud.tools.types import EvaluationResult

        env = Environment("test-env")

        @env.scenario("partial")
        async def partial_scenario():
            yield "Start the task"
            yield EvaluationResult(
                reward=0.3,
                done=False,  # Task not complete
                content="Partial progress",
            )

        await env.run_scenario_setup("partial", {})
        assert env._active_session is not None
        env._active_session.answer = "in progress"
        result = await env.run_scenario_evaluate("partial")

        assert result is not None
        assert result.reward == 0.3
        assert result.done is False


class TestPromptYieldTypes:
    """Test scenarios that yield different types for the prompt."""

    @pytest.mark.asyncio
    async def test_yield_text_content(self) -> None:
        """Scenario can yield TextContent for the prompt."""
        from mcp.types import TextContent

        env = Environment("test-env")

        @env.scenario("text-content")
        async def text_content_scenario():
            yield TextContent(text="Prompt from TextContent", type="text")
            yield 1.0

        prompt = await env.run_scenario_setup("text-content", {})
        assert prompt == "Prompt from TextContent"

    @pytest.mark.asyncio
    async def test_yield_list_of_strings(self) -> None:
        """Scenario can yield a list of strings (joined with newlines)."""
        env = Environment("test-env")

        @env.scenario("list-strings")
        async def list_strings_scenario():
            yield ["Line 1", "Line 2", "Line 3"]
            yield 1.0

        prompt = await env.run_scenario_setup("list-strings", {})
        assert prompt == "Line 1\nLine 2\nLine 3"

    @pytest.mark.asyncio
    async def test_yield_list_of_text_content(self) -> None:
        """Scenario can yield a list of TextContent blocks."""
        from mcp.types import TextContent

        env = Environment("test-env")

        @env.scenario("list-text-content")
        async def list_text_content_scenario():
            yield [
                TextContent(text="First part", type="text"),
                TextContent(text="Second part", type="text"),
            ]
            yield 1.0

        prompt = await env.run_scenario_setup("list-text-content", {})
        assert prompt == "First part\nSecond part"


class TestEvaluationResultDefaults:
    """Test EvaluationResult default behavior."""

    @pytest.mark.asyncio
    async def test_done_defaults_to_true(self) -> None:
        """EvaluationResult.done should default to True."""
        from hud.tools.types import EvaluationResult

        result = EvaluationResult(reward=0.5)
        assert result.done is True

    @pytest.mark.asyncio
    async def test_float_yield_implies_done(self) -> None:
        """Yielding a float should produce done=True."""
        env = Environment("test-env")

        @env.scenario("float-done")
        async def float_done_scenario():
            yield "Do something"
            yield 0.8  # Float yield

        await env.run_scenario_setup("float-done", {})
        assert env._active_session is not None
        env._active_session.answer = "done"
        result = await env.run_scenario_evaluate("float-done")

        assert result is not None
        assert result.reward == 0.8
        assert result.done is True  # Implied by float yield

    @pytest.mark.asyncio
    async def test_explicit_done_false(self) -> None:
        """Scenarios can explicitly set done=False for partial progress."""
        from hud.tools.types import EvaluationResult

        env = Environment("test-env")

        @env.scenario("partial-progress")
        async def partial_scenario():
            yield "Start task"
            yield EvaluationResult(reward=0.25, done=False)

        await env.run_scenario_setup("partial-progress", {})
        assert env._active_session is not None
        env._active_session.answer = "partial"
        result = await env.run_scenario_evaluate("partial-progress")

        assert result is not None
        assert result.done is False


class TestSubscoreUsage:
    """Test practical subscore usage patterns."""

    @pytest.mark.asyncio
    async def test_weighted_subscores(self) -> None:
        """Test subscores with different weights."""
        from hud.tools.types import EvaluationResult, SubScore

        env = Environment("test-env")

        @env.scenario("weighted")
        async def weighted_scenario():
            yield "Complete the task"
            # Weighted average: 0.6*1.0 + 0.3*0.5 + 0.1*0.0 = 0.75
            yield EvaluationResult(
                reward=0.75,
                done=True,
                subscores=[
                    SubScore(name="correctness", weight=0.6, value=1.0),
                    SubScore(name="efficiency", weight=0.3, value=0.5),
                    SubScore(name="style", weight=0.1, value=0.0),
                ],
            )

        await env.run_scenario_setup("weighted", {})
        assert env._active_session is not None
        env._active_session.answer = "result"
        result = await env.run_scenario_evaluate("weighted")

        assert result is not None
        assert result.reward == 0.75
        assert result.subscores is not None
        assert len(result.subscores) == 3
        # Verify subscores preserved order and values
        assert result.subscores[0].name == "correctness"
        assert result.subscores[0].value == 1.0
        assert result.subscores[2].name == "style"
        assert result.subscores[2].value == 0.0

    @pytest.mark.asyncio
    async def test_subscores_with_content(self) -> None:
        """Test subscores combined with explanation content."""
        from hud.tools.types import EvaluationResult, SubScore

        env = Environment("test-env")

        @env.scenario("explained")
        async def explained_scenario():
            yield "Evaluate this"
            yield EvaluationResult(
                reward=0.6,
                done=True,
                content="Found 3 of 5 items correctly",
                subscores=[
                    SubScore(name="detection", value=0.6),
                    SubScore(name="false_positives", value=1.0),  # Lower is better, inverted
                ],
            )

        await env.run_scenario_setup("explained", {})
        assert env._active_session is not None
        env._active_session.answer = "found 3 items"
        result = await env.run_scenario_evaluate("explained")

        assert result is not None
        assert result.content == "Found 3 of 5 items correctly"
        assert result.subscores is not None
        assert len(result.subscores) == 2


class TestNormalizationEdgeCases:
    """Test edge cases in yield normalization."""

    @pytest.mark.asyncio
    async def test_empty_string_prompt(self) -> None:
        """Empty string prompt should work."""
        env = Environment("test-env")

        @env.scenario("empty-prompt")
        async def empty_scenario():
            yield ""
            yield 1.0

        prompt = await env.run_scenario_setup("empty-prompt", {})
        assert prompt == ""

    @pytest.mark.asyncio
    async def test_zero_reward(self) -> None:
        """Zero reward should work correctly."""
        env = Environment("test-env")

        @env.scenario("zero-reward")
        async def zero_scenario():
            yield "Try something"
            yield 0.0

        await env.run_scenario_setup("zero-reward", {})
        assert env._active_session is not None
        env._active_session.answer = "failed"
        result = await env.run_scenario_evaluate("zero-reward")

        assert result is not None
        assert result.reward == 0.0
        assert result.done is True

    @pytest.mark.asyncio
    async def test_negative_reward(self) -> None:
        """Negative reward (penalty) should work."""
        from hud.tools.types import EvaluationResult

        env = Environment("test-env")

        @env.scenario("penalty")
        async def penalty_scenario():
            yield "Don't break anything"
            yield EvaluationResult(reward=-0.5, done=True, content="Caused damage")

        await env.run_scenario_setup("penalty", {})
        assert env._active_session is not None
        env._active_session.answer = "broke it"
        result = await env.run_scenario_evaluate("penalty")

        assert result is not None
        assert result.reward == -0.5

    @pytest.mark.asyncio
    async def test_reward_above_one(self) -> None:
        """Reward above 1.0 (bonus) should work."""
        env = Environment("test-env")

        @env.scenario("bonus")
        async def bonus_scenario():
            yield "Do extra well"
            yield 1.5  # Exceptional performance

        await env.run_scenario_setup("bonus", {})
        assert env._active_session is not None
        env._active_session.answer = "exceeded"
        result = await env.run_scenario_evaluate("bonus")

        assert result is not None
        assert result.reward == 1.5
