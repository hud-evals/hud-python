"""Tests for hud.cli.eval module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from mcp import types

from hud.cli.eval import (
    build_agent,
    run_single_task,
)
from hud.types import AgentType, Task, Trace


class TestBuildAgent:
    """Test the build_agent function."""

    def test_builds_integration_test_agent(self) -> None:
        """
        Test building an integration test agent.
        """
        with patch("hud.agents.misc.integration_test_agent.IntegrationTestRunner") as mock_runner:
            mock_instance = Mock()
            mock_runner.return_value = mock_instance

            # Test with verbose=False
            result = build_agent(AgentType.INTEGRATION_TEST, verbose=False)

            mock_runner.assert_called_once_with(verbose=False)
            assert result == mock_instance

    def test_builds_claude_agent(self) -> None:
        """
        Test building a Claude agent with default model.
        """
        with patch("hud.agents.ClaudeAgent") as mock_runner:
            mock_instance = Mock()
            mock_runner.return_value = mock_instance

            # Test with verbose=False
            result = build_agent(AgentType.CLAUDE, verbose=False)

            mock_runner.assert_called_once_with(model="claude-sonnet-4-20250514", verbose=False)
            assert result == mock_instance

    def test_builds_claude_agent_with_custom_model_and_allowed_tools(self) -> None:
        """
        Test building a Claude agent with custom model name and allowed tools.
        """
        with patch("hud.agents.ClaudeAgent") as mock_runner:
            mock_instance = Mock()
            mock_runner.return_value = mock_instance

            # Test with verbose=False
            result = build_agent(
                AgentType.CLAUDE,
                model="claude-sonnet-4-20250514",
                allowed_tools=["act"],
                verbose=True,
            )

            mock_runner.assert_called_once_with(
                model="claude-sonnet-4-20250514",
                allowed_tools=["act"],
                verbose=True,
            )
            assert result == mock_instance


class TestRunSingleTask:
    """Test the run_single_task function."""

    @pytest.mark.asyncio
    async def test_applies_agent_config_from_task(self) -> None:
        """Test that task.agent_config is applied during agent initialization."""
        mock_task = Task(
            prompt="Test",
            mcp_config={"local": {"url": "http://localhost:8765/mcp"}},
            agent_config={
                "system_prompt": "Custom instructions",
                "allowed_tools": ["tool1", "tool2"],
                "append_setup_output": False,
            },
        )
        mock_agent = AsyncMock(
            initialize=AsyncMock(), run=AsyncMock(return_value=Trace(reward=1.0, done=True))
        )

        with (
            patch("hud.utils.tasks.load_tasks", return_value=[mock_task]),
            patch(
                "hud.agents.misc.integration_test_agent.IntegrationTestRunner",
                return_value=mock_agent,
            ),
            patch("hud.cli.eval.find_environment_dir", return_value=None),
            patch("hud.cli.eval.hud.trace"),
        ):
            await run_single_task("test.json", agent_type=AgentType.INTEGRATION_TEST, max_steps=10)

            # Verify agent.run was called with the task containing agent_config
            mock_agent.run.assert_called_once()
            called_task = mock_agent.run.call_args[0][0]
            assert called_task.agent_config == mock_task.agent_config

    @pytest.mark.asyncio
    async def test_runs_with_group_size_greater_than_one(self) -> None:
        """Test that group_size > 1 triggers run_tasks_grouped instead of agent.run."""
        mock_task = Task(prompt="Test", mcp_config={"local": {"url": "http://localhost:8765/mcp"}})

        with (
            patch("hud.utils.tasks.load_tasks", return_value=[mock_task]),
            patch("hud.cli.eval.run_tasks_grouped", new_callable=AsyncMock) as mock_grouped,
            patch("hud.cli.eval.display_group_statistics"),
            patch("hud.cli.eval.find_environment_dir", return_value=None),
            patch("hud.cli.eval.hud.trace"),
        ):
            mock_grouped.return_value = [{"task": mock_task, "rewards": [1.0, 0.5]}]

            await run_single_task(
                "test.json", agent_type=AgentType.INTEGRATION_TEST, group_size=3, max_steps=10
            )

            # Verify run_tasks_grouped was called with correct group_size
            mock_grouped.assert_called_once()
            assert mock_grouped.call_args.kwargs["group_size"] == 3
            assert mock_grouped.call_args.kwargs["max_steps"] == 10


class TestToolFiltering:
    """Test wildcard tool filtering via agent_config in tasks."""

    @pytest.fixture
    def mock_mcp_client(self):
        """Fixture for mock MCP client."""
        client = MagicMock()
        client.initialize = AsyncMock()
        client.mcp_config = {"local": {"url": "http://localhost"}}
        return client

    @pytest.fixture
    def mock_model_client(self):
        """Fixture for mock Anthropic client."""
        return MagicMock()

    async def _run_agent_with_tools(
        self,
        mock_mcp_client: MagicMock,
        mock_model_client: MagicMock,
        tools: list[types.Tool],
        agent_config: dict | None = None,
    ) -> list[types.Tool]:
        """Helper to create agent, initialize with tools and config, return filtered tools."""
        from hud.agents import ClaudeAgent

        mock_mcp_client.list_tools = AsyncMock(return_value=tools)

        task = Task(
            prompt="Test",
            mcp_config={"local": {"url": "http://localhost"}},
            agent_config=agent_config or {},
        )

        agent = ClaudeAgent(
            mcp_client=mock_mcp_client,
            model_client=mock_model_client,
            model="test",
            validate_api_key=False,
        )
        await agent.initialize(task)
        return agent.get_available_tools()

    @pytest.mark.asyncio
    async def test_no_filters_returns_all_tools(self, mock_mcp_client, mock_model_client) -> None:
        """Test that no filters in agent_config returns all tools."""
        tools = [
            types.Tool(name="tool1", description="Tool 1", inputSchema={}),
            types.Tool(name="tool2", description="Tool 2", inputSchema={}),
            types.Tool(name="debug_tool", description="Debug", inputSchema={}),
        ]

        result = await self._run_agent_with_tools(mock_mcp_client, mock_model_client, tools)

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_allowed_tools_filters_correctly(
        self, mock_mcp_client, mock_model_client
    ) -> None:
        """Test that allowed_tools in agent_config filters to matching patterns."""
        tools = [
            types.Tool(name="screenshot_take", description="Tool 1", inputSchema={}),
            types.Tool(name="screenshot_full", description="Tool 2", inputSchema={}),
            types.Tool(name="click", description="Tool 3", inputSchema={}),
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
            types.Tool(name="tool1", description="Tool 1", inputSchema={}),
            types.Tool(name="debug_tool", description="Tool 2", inputSchema={}),
            types.Tool(name="internal_secret", description="Tool 3", inputSchema={}),
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
            types.Tool(name="browser_click", description="Tool 1", inputSchema={}),
            types.Tool(name="browser_debug", description="Tool 2", inputSchema={}),
            types.Tool(name="system_click", description="Tool 3", inputSchema={}),
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
            types.Tool(name="browser_click", description="Click", inputSchema={}),
            types.Tool(name="browser_type", description="Type", inputSchema={}),
            types.Tool(name="browser_debug", description="Debug", inputSchema={}),
            types.Tool(name="system_screenshot", description="Screenshot", inputSchema={}),
            types.Tool(name="system_execute", description="Execute", inputSchema={}),
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


class TestSystemPromptHandling:
    """Test system prompt handling through run_dataset flow."""

    @pytest.fixture
    def mock_mcp_client(self):
        """Fixture for mock MCP client."""
        client = MagicMock()
        client.initialize = AsyncMock()
        client.list_tools = AsyncMock(return_value=[])
        client.shutdown = AsyncMock()
        client.mcp_config = {"local": {"url": "http://localhost:8765/mcp"}}
        return client

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
        from hud.agents.base import GLOBAL_SYSTEM_PROMPT
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
        agent_init_config = {
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

            # Verify the task system prompt was appended
            assert captured_agent.system_prompt.endswith(f"\n\n{task_system_prompt}")
            # Verify it starts with the base global system prompt
            assert captured_agent.system_prompt.startswith(GLOBAL_SYSTEM_PROMPT)

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
