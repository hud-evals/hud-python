"""Tests for hud.cli.eval module."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic import AsyncAnthropic
from mcp import types

from hud.agents.tests.conftest import MockMCPClient
from hud.types import Task, Trace


class TestToolFiltering:
    """Test wildcard tool filtering via agent_config in tasks."""

    @pytest.fixture
    def mock_mcp_client(self):
        """Fixture for mock MCP client."""
        return MockMCPClient()

    @pytest.fixture
    def mock_model_client(self):
        """Fixture for a lightweight Anthropic client."""
        client = AsyncAnthropic(api_key="test_key")
        client.__dict__["beta"] = SimpleNamespace(messages=AsyncMock())
        return client

    async def _run_agent_with_tools(
        self,
        mock_mcp_client: MagicMock,
        mock_model_client: MagicMock,
        tools: list[types.Tool],
        agent_config: dict[str, Any] | None = None,
    ) -> list[types.Tool]:
        """Helper to create agent, initialize with tools and config, return filtered tools."""
        from hud.agents import ClaudeAgent
        from hud.types import BaseAgentConfig

        mock_mcp_client.list_tools = AsyncMock(return_value=tools)

        task = Task(
            prompt="Test",
            mcp_config={"local": {"url": "http://localhost"}},
            agent_config=BaseAgentConfig(**agent_config) if agent_config else None,
        )

        agent = ClaudeAgent.create(
            mcp_client=mock_mcp_client,
            model_client=mock_model_client,
            checkpoint_name="test",
            validate_api_key=False,
        )
        await agent.initialize(task)
        return agent.get_available_tools()

    @pytest.mark.asyncio
    async def test_no_filters_returns_all_tools(self, mock_mcp_client, mock_model_client) -> None:
        """Test that no filters in agent_config returns all tools."""
        tools = [
            types.Tool(
                name="tool1",
                description="Tool 1",
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="tool2",
                description="Tool 2",
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="debug_tool",
                description="Debug",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

        result = await self._run_agent_with_tools(mock_mcp_client, mock_model_client, tools)

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_allowed_tools_filters_correctly(
        self, mock_mcp_client, mock_model_client
    ) -> None:
        """Test that allowed_tools in agent_config filters to matching patterns."""
        tools = [
            types.Tool(
                name="screenshot_take",
                description="Tool 1",
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="screenshot_full",
                description="Tool 2",
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="click",
                description="Tool 3",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]
        agent_config = {"allowed_tools": ["screenshot_*"]}

        result = await self._run_agent_with_tools(
            mock_mcp_client, mock_model_client, tools, agent_config
        )

        assert len(result) == 2
        assert all("screenshot" in t.name for t in result)

    @pytest.mark.asyncio
    async def test_disallowed_tools_excludes_correctly(
        self, mock_mcp_client, mock_model_client
    ) -> None:
        """Test that disallowed_tools in agent_config excludes matching patterns."""
        tools = [
            types.Tool(
                name="tool1",
                description="Tool 1",
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="debug_tool",
                description="Tool 2",
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="internal_secret",
                description="Tool 3",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]
        agent_config = {"disallowed_tools": ["debug_*", "internal_*"]}

        result = await self._run_agent_with_tools(
            mock_mcp_client, mock_model_client, tools, agent_config
        )

        assert len(result) == 1
        assert result[0].name == "tool1"

    @pytest.mark.asyncio
    async def test_both_filters_applies_allowed_then_disallowed(
        self, mock_mcp_client, mock_model_client
    ) -> None:
        """Test that both filters in agent_config work together (disallowed takes precedence)."""
        tools = [
            types.Tool(
                name="browser_click",
                description="Tool 1",
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="browser_debug",
                description="Tool 2",
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="system_click",
                description="Tool 3",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]
        agent_config = {"allowed_tools": ["browser_*"], "disallowed_tools": ["*_debug"]}

        result = await self._run_agent_with_tools(
            mock_mcp_client, mock_model_client, tools, agent_config
        )

        assert len(result) == 1
        assert result[0].name == "browser_click"


class TestRunDatasetToolFiltering:
    """Test tool filtering via run_dataset with agent_config in both init and task."""

    @pytest.fixture
    def all_tools(self):
        """Fixture for a standard set of tools."""
        return [
            types.Tool(
                name="browser_click",
                description="Click",
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="browser_type",
                description="Type",
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="browser_debug",
                description="Debug",
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="system_screenshot",
                description="Screenshot",
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="system_execute",
                description="Execute",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

    @pytest.fixture
    def captured_agent_fixture(self):
        """Fixture that returns a dictionary to capture the agent instance."""
        return {"agent": None}

    @pytest.fixture
    def mock_run_context(self, captured_agent_fixture):
        """Fixture for mocking _run_context."""

        async def _mock(self, context, max_steps=10):
            captured_agent_fixture["agent"] = self
            return Trace(reward=1.0, done=True, content="Done")

        return _mock

    @pytest.fixture
    def mock_call_tools(self):
        """Fixture for mocking call_tools."""

        async def _mock(self, tool_call=None):
            return []

        return _mock

    @pytest.fixture
    def mock_client_instance(self, all_tools):
        """Fixture for mock MCP client instance."""
        mock_client = MagicMock()
        mock_client.initialize = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=all_tools)
        mock_client.shutdown = AsyncMock()
        mock_client.mcp_config = {"local": {"url": "http://localhost:8765/mcp"}}
        return mock_client

    @pytest.mark.asyncio
    async def test_agent_config_intersection_union_via_run_dataset(
        self,
        all_tools,
        captured_agent_fixture,
        mock_run_context,
        mock_call_tools,
        mock_client_instance,
    ) -> None:
        """Test that allowed_tools intersect and disallowed_tools union when set in both __init__ and task.agent_config."""  # noqa: E501
        from hud.agents import ClaudeAgent
        from hud.datasets.runner import run_dataset

        # Create a task with its own agent_config
        task_dict = {
            "prompt": "Test task",
            "mcp_config": {"local": {"url": "http://localhost:8765/mcp"}},
            "agent_config": {
                "allowed_tools": [
                    "browser_*",
                    "system_screenshot",
                ],  # Task wants browser_* and system_screenshot
                "disallowed_tools": [
                    "*_debug",
                    "*_execute",
                ],  # Task disallows *_debug and *_execute
            },
        }

        # Agent config passed to __init__ via run_dataset
        agent_init_config = {
            "allowed_tools": ["browser_*", "system_*"],  # Agent init wants browser_* and system_*
            "disallowed_tools": ["browser_debug"],  # Agent init disallows browser_debug
            "validate_api_key": False,
        }

        with (
            patch("hud.job"),
            patch("hud.trace"),
            patch.object(ClaudeAgent, "_run_context", mock_run_context),
            patch.object(ClaudeAgent, "call_tools", mock_call_tools),
            patch("hud.clients.MCPClient", return_value=mock_client_instance),
            patch("hud.settings.settings.anthropic_api_key", "sk-test-key"),
        ):
            # Run the dataset
            await run_dataset(
                name="test_job",
                dataset=[task_dict],
                agent_class=ClaudeAgent,
                agent_config=agent_init_config,
                max_steps=10,
            )

            # Verify agent was created and ran
            captured_agent = captured_agent_fixture["agent"]
            assert captured_agent is not None

            # Get the filtered tools
            filtered_tools = captured_agent.get_available_tools()
            filtered_names = {tool.name for tool in filtered_tools}

            # Expected behavior:
            # 1. allowed_tools intersection: ["browser_*", "system_*"] ∩ ["browser_*", "system_screenshot"] # noqa: E501
            #    Exact string intersection: only "browser_*" is in both lists
            #    So only tools matching browser_* are allowed: browser_click, browser_type, browser_debug # noqa: E501
            # 2. disallowed_tools union: ["browser_debug"] U ["*_debug", "*_execute"]
            #    Result: ["browser_debug", "*_debug", "*_execute"] (all patterns included)
            # 3. Final: {browser_click, browser_type, browser_debug} - {browser_debug}
            #    Result: browser_click, browser_type

            expected_tools = {"browser_click", "browser_type"}
            assert filtered_names == expected_tools, (
                f"Expected {expected_tools}, got {filtered_names}"
            )

    @pytest.mark.asyncio
    async def test_no_allowed_tools_keeps_all_tools_except_disallowed(
        self,
        all_tools,
        captured_agent_fixture,
        mock_run_context,
        mock_call_tools,
        mock_client_instance,
    ) -> None:
        """Test that when allowed_tools is not set, all tools are available except disallowed ones."""  # noqa: E501
        from hud.agents import ClaudeAgent
        from hud.datasets.runner import run_dataset

        # Create a task with its own agent_config (no allowed_tools)
        task_dict = {
            "prompt": "Test task",
            "mcp_config": {"local": {"url": "http://localhost:8765/mcp"}},
            "agent_config": {
                # No allowed_tools set - should allow all tools
                "disallowed_tools": ["*_execute"],  # Task disallows *_execute
            },
        }

        # Agent config passed to __init__ via run_dataset (no allowed_tools)
        agent_init_config = {
            # No allowed_tools set - should allow all tools
            "disallowed_tools": ["browser_debug"],  # Agent init disallows browser_debug
            "validate_api_key": False,
        }

        with (
            patch("hud.job"),
            patch("hud.trace"),
            patch.object(ClaudeAgent, "_run_context", mock_run_context),
            patch.object(ClaudeAgent, "call_tools", mock_call_tools),
            patch("hud.clients.MCPClient", return_value=mock_client_instance),
            patch("hud.settings.settings.anthropic_api_key", "sk-test-key"),
        ):
            # Run the dataset
            await run_dataset(
                name="test_job",
                dataset=[task_dict],
                agent_class=ClaudeAgent,
                agent_config=agent_init_config,
                max_steps=10,
            )

            # Verify agent was created and ran
            captured_agent = captured_agent_fixture["agent"]
            assert captured_agent is not None

            # Get the filtered tools
            filtered_tools = captured_agent.get_available_tools()
            filtered_names = {tool.name for tool in filtered_tools}

            # Expected behavior:
            # 1. allowed_tools: None (no allowed_tools set in either init or task)
            #    Result: All tools are initially allowed
            # 2. disallowed_tools union: ["browser_debug"] U ["*_execute"]
            #    Result: ["browser_debug", "*_execute"] (all patterns included)
            # 3. Final: {all tools} - {browser_debug, system_execute}
            #    Result: browser_click, browser_type, system_screenshot

            expected_tools = {"browser_click", "browser_type", "system_screenshot"}
            assert filtered_names == expected_tools, (
                f"Expected {expected_tools}, got {filtered_names}"
            )


SYSTEM_PROMPT = "You are an assistant that can use tools to help the user. You will be given a task and you will need to use the tools to complete the task."  # noqa: E501


class TestSystemPromptHandling:
    """Test system prompt handling through run_dataset flow."""

    @pytest.fixture
    def mock_mcp_client(self):
        """Fixture for mock MCP client."""
        return MockMCPClient()

    @pytest.fixture
    def captured_agent_fixture(self):
        """Fixture that returns a dictionary to capture the agent instance."""
        return {"agent": None}

    @pytest.fixture
    def mock_run_context(self, captured_agent_fixture):
        """Fixture for mocking _run_context to capture agent."""

        async def _mock(self, context, max_steps=10):
            captured_agent_fixture["agent"] = self
            return Trace(reward=1.0, done=True, content="Done")

        return _mock

    @pytest.fixture
    def mock_call_tools(self):
        """Fixture for mocking call_tools."""

        async def _mock(self, tool_call=None):
            return []

        return _mock

    @pytest.mark.asyncio
    async def test_task_system_prompt_only(
        self, captured_agent_fixture, mock_run_context, mock_call_tools, mock_mcp_client
    ) -> None:
        """Test that task system_prompt is appended when agent has default system prompt."""
        from hud.agents import ClaudeAgent
        from hud.datasets.runner import run_dataset

        task_system_prompt = "Task prompt"

        # Create a task with its own system_prompt in agent_config
        task_dict = {
            "prompt": "Test task",
            "mcp_config": {"local": {"url": "http://localhost:8765/mcp"}},
            "agent_config": {
                "system_prompt": task_system_prompt,
            },
        }

        # Agent config with no custom system_prompt (will use default)
        agent_init_config = {"validate_api_key": False, "system_prompt": SYSTEM_PROMPT}

        with (
            patch("hud.job"),
            patch("hud.trace"),
            patch.object(ClaudeAgent, "_run_context", mock_run_context),
            patch.object(ClaudeAgent, "call_tools", mock_call_tools),
            patch("hud.clients.MCPClient", return_value=mock_mcp_client),
            patch("hud.settings.settings.anthropic_api_key", "sk-test-key"),
        ):
            # Run the dataset
            await run_dataset(
                name="test_job",
                dataset=[task_dict],
                agent_class=ClaudeAgent,
                agent_config=agent_init_config,
                max_steps=10,
            )

            # Verify agent was created and ran
            captured_agent = captured_agent_fixture["agent"]
            assert captured_agent is not None

            # Verify the task system prompt was appended
            assert captured_agent.system_prompt.endswith(f"\n\n{task_system_prompt}")
            # Verify it starts with the base global system prompt
            assert captured_agent.system_prompt.startswith(SYSTEM_PROMPT)

    @pytest.mark.asyncio
    async def test_both_agent_and_task_system_prompts(
        self, captured_agent_fixture, mock_run_context, mock_call_tools, mock_mcp_client
    ) -> None:
        """Test that both agent init and task system prompts are present when both are set."""
        from hud.agents import ClaudeAgent
        from hud.datasets.runner import run_dataset

        agent_custom_prompt = "Agent init prompt"
        task_system_prompt = "Task prompt"

        # Create a task with its own system_prompt in agent_config
        task_dict = {
            "prompt": "Test task",
            "mcp_config": {"local": {"url": "http://localhost:8765/mcp"}},
            "agent_config": {
                "system_prompt": task_system_prompt,
            },
        }

        # Agent config WITH custom system_prompt
        agent_init_config = {
            "system_prompt": agent_custom_prompt,
            "validate_api_key": False,
        }

        with (
            patch("hud.job"),
            patch("hud.trace"),
            patch.object(ClaudeAgent, "_run_context", mock_run_context),
            patch.object(ClaudeAgent, "call_tools", mock_call_tools),
            patch("hud.clients.MCPClient", return_value=mock_mcp_client),
            patch("hud.settings.settings.anthropic_api_key", "sk-test-key"),
        ):
            # Run the dataset
            await run_dataset(
                name="test_job",
                dataset=[task_dict],
                agent_class=ClaudeAgent,
                agent_config=agent_init_config,
                max_steps=10,
            )

            # Verify agent was created and ran
            captured_agent = captured_agent_fixture["agent"]
            assert captured_agent is not None

            # Verify the task system prompt was appended at the end
            assert captured_agent.system_prompt.endswith(f"\n\n{task_system_prompt}")
            # Verify it starts with the agent custom prompt
            assert captured_agent.system_prompt.startswith(agent_custom_prompt)
            # Verify both prompts are present
            assert agent_custom_prompt in captured_agent.system_prompt
            assert task_system_prompt in captured_agent.system_prompt
