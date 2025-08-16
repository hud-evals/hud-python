"""Base MCP Agent implementation."""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import mcp.types as types

from .types import AgentResponse, MCPToolCall, MCPToolResult, Trace

if TYPE_CHECKING:
    from hud.datasets import TaskConfig

    from .clients.base import AgentMCPClient


logger = logging.getLogger(__name__)


class MCPAgent(ABC):
    """
    Base class for MCP-enabled agents.

    This class provides the foundation for agents that interact with MCP servers,
    handling tool discovery and filtering while leaving provider-specific
    implementation details to subclasses.
    """

    def __init__(
        self,
        mcp_client: AgentMCPClient | None = None,
        allowed_tools: list[str] | None = None,
        disallowed_tools: list[str] | None = None,
        initial_screenshot: bool = False,
        max_screenshot_history: int = 3,
        append_tool_system_prompt: bool = True,
        custom_system_prompt: str | None = None,
        lifecycle_tools: list[str] | None = None,
        append_setup_content: bool = False,
    ) -> None:
        """
        Initialize the base MCP agent.

        Args:
            mcp_client: AgentMCPClient instance for server connections
            allowed_tools: List of tool names to allow (None = all tools)
            disallowed_tools: List of tool names to disallow
            initial_screenshot: Whether to capture screenshot before first prompt
            max_screenshot_history: Maximum number of screenshots to keep in context
            append_tool_system_prompt: Whether to append available tools to system prompt
            custom_system_prompt: Custom system prompt to use
            lifecycle_tools: List of tool names to use for lifecycle tools
        """
        if not mcp_client:
            raise ValueError(
                "MCPClient is required. Please provide a configured MCPClient instance."
            )
        self.mcp_client = mcp_client

        # Filtering
        self.allowed_tools = allowed_tools
        self.disallowed_tools = disallowed_tools or []

        # Screenshots
        self.initial_screenshot = initial_screenshot
        self.max_screenshot_history = max_screenshot_history
        self.append_tool_system_prompt = append_tool_system_prompt
        self.custom_system_prompt = custom_system_prompt
        self.append_setup_content = append_setup_content

        self.lifecycle_tools = lifecycle_tools or []

        self.model_name = "test-agent"

        # Initialize these here so methods can be called before initialize()
        self._available_tools: list[types.Tool] = []
        self._tool_map: dict[str, types.Tool] = {}  # Simplified: just name to tool
        self.screenshot_history: list[str] = []

    async def _filter_tools(self) -> None:
        """Apply tool filtering based on allowed/disallowed lists."""
        # Get all tools from client
        all_tools = await self.mcp_client.list_tools()

        # Filter tools
        self._available_tools = []
        self._tool_map = {}

        for tool in all_tools:
            # Check if tool should be included
            if self.allowed_tools and tool.name not in self.allowed_tools:
                continue
            if tool.name in self.disallowed_tools:
                continue

            self._available_tools.append(tool)
            # Simplified mapping - just tool name to tool
            self._tool_map[tool.name] = tool

    async def initialize(self, task: str | TaskConfig | None = None) -> None:
        """Initialize the agent with task-specific configuration."""
        # Initialize client if needed
        await self.mcp_client.initialize()

        # If task is provided, add lifecycle tools
        from hud.datasets import TaskConfig

        if isinstance(task, TaskConfig):
            if task.setup_tool:
                if isinstance(task.setup_tool, list):
                    for tool in task.setup_tool:
                        self.lifecycle_tools.append(tool.name)
                else:
                    self.lifecycle_tools.append(task.setup_tool.name)
            if task.evaluate_tool:
                if isinstance(task.evaluate_tool, list):
                    for tool in task.evaluate_tool:
                        self.lifecycle_tools.append(tool.name)
                else:
                    self.lifecycle_tools.append(task.evaluate_tool.name)

        # Re-apply filtering with updated lifecycle tools
        await self._filter_tools()

        logger.info(
            "Agent initialized with %d available tools (after filtering)",
            len(self._available_tools),
        )

    def get_available_tools(self) -> list[types.Tool]:
        """Get list of available MCP tools for LLM use (excludes lifecycle tools)."""
        lifecycle_tool_names = self.lifecycle_tools
        return [tool for tool in self._available_tools if tool.name not in lifecycle_tool_names]

    def get_system_prompt(self) -> str:
        """Generate system prompt with optional tool information."""
        base_prompt = self.custom_system_prompt or "You are a helpful assistant."

        if self.append_tool_system_prompt and self._available_tools:
            tool_descriptions = []
            for tool in self._available_tools:
                desc = f"- {tool.name}: {tool.description}"
                if tool.inputSchema:
                    desc += f" (parameters: {tool.inputSchema})"
                tool_descriptions.append(desc)

            tools_prompt = "\n\nYou have access to the following tools:\n" + "\n".join(
                tool_descriptions
            )
            return base_prompt + tools_prompt

        return base_prompt

    async def call_tool(self, tool_call: MCPToolCall | None = None) -> MCPToolResult:
        """
        Call a tool through the MCP client.

        Args:
            tool_call: Dict with 'name' and optional 'arguments' keys

        Returns:
            The raw MCPToolResult
        """
        if tool_call is None:
            raise ValueError("tool_call must be an MCPToolCall object")

        tool_name = tool_call.name
        if not tool_name:
            raise ValueError("Tool call must have a 'name' field")

        tool_args = tool_call.arguments

        if tool_name not in self._tool_map and tool_name not in self.lifecycle_tools:
            raise ValueError(f"Tool '{tool_name}' not found or not allowed")

        if self.mcp_client is None:
            raise ValueError("Client is not initialized")

        # Use client's call_tool method which handles routing
        result = await self.mcp_client.call_tool(tool_name, tool_args)

        # Log result for debugging
        if result.isError:
            logger.error("Tool '%s' returned error: %s", tool_name, result.content)
        else:
            logger.debug("Tool '%s' completed successfully", tool_name)

        return result

    async def execute_tools(
        self, tool_calls: MCPToolCall | list[MCPToolCall]
    ) -> list[MCPToolResult]:
        """Execute a list of tools with error handling."""
        if isinstance(tool_calls, MCPToolCall):
            tool_calls = [tool_calls]

        results: list[MCPToolResult] = []
        for tool_call in tool_calls:
            try:
                logger.info("Calling tool: %s", tool_call)
                results.append(await self.call_tool(tool_call))
            except Exception as e:
                logger.error("Tool execution failed: %s", e)
                results.append(self._format_error_result(str(e)))
        return results

    def has_computer_tools(self) -> bool:
        """Check if any computer control tools are available."""
        computer_tools = {"computer", "computer_anthropic", "computer_openai", "screenshot"}
        return any(tool.name in computer_tools for tool in self._available_tools)

    def get_tool_schemas(self) -> list[dict]:
        """Get tool schemas in a format suitable for the model."""
        schemas = []
        for tool in self._available_tools:
            # Filter out lifecycle tools from LLM conversation
            if tool.name in self.lifecycle_tools:
                continue

            schema = {
                "name": tool.name,
                "description": tool.description,
            }
            if tool.inputSchema:
                schema["parameters"] = tool.inputSchema
            schemas.append(schema)
        return schemas

    async def capture_screenshot(self) -> str | None:
        """Capture a screenshot using available tools."""
        if not self.has_computer_tools():
            return None

        # Try different screenshot tools
        for tool_name in [
            "computer",
            "screenshot",
            "computer_anthropic",
            "computer_openai",
            "anthropic_computer",
            "openai_computer",
        ]:
            if tool_name in self._tool_map:
                try:
                    # Different tools have different APIs
                    if tool_name == "computer_openai":
                        tool_call = MCPToolCall(name=tool_name, arguments={"type": "screenshot"})
                    else:
                        tool_call = MCPToolCall(name=tool_name, arguments={"action": "screenshot"})

                    result = await self.call_tool(tool_call)

                    # Extract screenshot from result
                    for content in result.content:
                        if isinstance(content, types.ImageContent):
                            logger.info("Captured screenshot")
                            return content.data

                except Exception as e:
                    logger.warning("Failed to capture screenshot with %s: %s", tool_name, e)

        return None

    def extract_latest_screenshot(self, tool_results: list[MCPToolResult]) -> str | None:
        """Extract the latest screenshot from tool results."""
        latest_screenshot = None
        for result in tool_results:
            if not result.isError:
                for content in result.content:
                    if isinstance(content, types.ImageContent):
                        latest_screenshot = content.data
        return latest_screenshot

    async def run(self, prompt_or_task: str | TaskConfig, max_steps: int = 10) -> Trace:
        """
        Run the agent with the given prompt or task.

        Args:
            prompt_or_task: Either a string prompt for simple execution or a Task object
            max_steps: Maximum number of steps (-1 for infinite)

        Returns:
            Trace with reward, done, content, isError fields and trace steps
        """
        # Import here to avoid circular imports
        from hud.datasets import TaskConfig

        if len(self._available_tools) == 0:
            await self.initialize(prompt_or_task)

        # Handle Task objects with full lifecycle
        if isinstance(prompt_or_task, TaskConfig):
            return await self._run_task(prompt_or_task, max_steps)

        # Handle simple string prompts
        elif isinstance(prompt_or_task, str):
            return await self.run_prompt(prompt_or_task, max_steps=max_steps)

        else:
            raise TypeError(f"prompt_or_task must be str or TaskConfig, got {type(prompt_or_task)}")

    async def _run_task(self, task: TaskConfig, max_steps: int = 10) -> Trace:
        """
        Execute a task with setup and evaluate phases.

        Args:
            task: Task object with prompt, setup, and evaluate configs
            max_steps: Maximum steps for task execution (-1 for infinite)

        Returns:
            Trace with reward from evaluation
        """
        prompt_result = None

        try:
            # Setup phase
            start_prompt = ""
            if task.setup_tool is not None:
                logger.info("Setting up tool phase: %s", task.setup_tool)
                results = await self.execute_tools(task.setup_tool)
                if any(result.isError for result in results):
                    raise RuntimeError(f"{results}")

                if self.append_setup_content and isinstance(results[0].content, list):
                    for content in results[0].content:
                        if isinstance(content, types.TextContent):
                            start_prompt += f"{content.text}\n"

            # Initialize conversation with the prompt and setup content
            start_prompt += task.prompt
            logger.info("Start prompt: %s", start_prompt)

            # Execute the task
            prompt_result = await self.run_prompt(start_prompt, max_steps=max_steps)

        except Exception as e:
            logger.error("Task execution failed: %s", e)
            # Create an error result but don't return yet - we still want to evaluate
            prompt_result = Trace(reward=0.0, done=True, content=str(e), isError=True)
            prompt_result.populate_from_context()

        # Always evaluate if we have a prompt result and evaluate tool
        if prompt_result is not None and task.evaluate_tool is not None:
            try:
                logger.info("Evaluating tool phase: %s", task.evaluate_tool)
                results = await self.execute_tools(task.evaluate_tool)

                if any(result.isError for result in results):
                    raise RuntimeError(f"{results}")

                # Extract reward and content from evaluation
                if results:
                    reward = _find_reward(results[0])
                    eval_content = _find_content(results[0])

                    # Update the prompt result with evaluation reward
                    prompt_result.reward = reward
                    if eval_content:
                        prompt_result.content = eval_content

            except Exception as e:
                logger.error("Evaluation phase failed: %s", e)
                # Continue with the prompt result even if evaluation failed

        return (
            prompt_result
            if prompt_result
            else Trace(reward=0.0, done=True, content="No result available", isError=True)
        )

    def _format_error_result(self, error_message: str) -> MCPToolResult:
        return MCPToolResult(
            content=[types.TextContent(text=error_message, type="text")], isError=True
        )

    async def run_prompt(self, prompt: str, *, max_steps: int = 10) -> Trace:
        """
        Run the agent with the given prompt. This is the core agent loop.

        Args:
            prompt: The prompt to complete
            max_steps: Maximum number of steps (-1 for infinite)

        Returns:
            Trace with reward, done, content fields and trace steps
        """
        final_response = None
        error = None

        try:
            # Initialize conversation with the prompt
            messages = await self.create_initial_messages(prompt, self.initial_screenshot)
            logger.info("Messages: %s", messages)

            step_count = 0
            while max_steps == -1 or step_count < max_steps:
                step_count += 1
                if max_steps == -1:
                    logger.info("Step %s (unlimited)", step_count)
                else:
                    logger.info("Step %s/%s", step_count, max_steps)

                try:
                    # 1. Get model response
                    response = await self.get_model_response(messages)

                    logger.info("Agent:\n%s", response)

                    # Check if we should stop
                    if response.done or not response.tool_calls:
                        logger.info(
                            "Agent finished - done=%s, tool_calls=%s",
                            response.done,
                            len(response.tool_calls),
                        )
                        final_response = response
                        break

                    # 2. Execute tools
                    tool_calls = response.tool_calls
                    tool_results = await self.execute_tools(tool_calls)

                    # 3. Format tool results and add to messages
                    tool_messages = await self.format_tool_results(tool_calls, tool_results)
                    messages.extend(tool_messages)

                except Exception as e:
                    logger.error("Step failed: %s", e)
                    error = str(e)
                    break

        except KeyboardInterrupt:
            logger.info("Agent execution interrupted by user")
            error = "Interrupted by user"
        except asyncio.CancelledError:
            logger.info("Agent execution cancelled")
            error = "Cancelled"
        except Exception as e:
            logger.error("Unexpected error: %s", e)
            error = str(e)

        # Build result
        trace_result = Trace(
            reward=0.0,  # Default - will be set by task evaluation if applicable
            done=True,
            content=final_response.content if final_response else None,
            isError=error is not None,
            info={"error": error} if error else {},
        )

        # Populate trace steps from current context
        trace_result.populate_from_context()

        return trace_result

    @abstractmethod
    async def create_initial_messages(
        self, prompt: str, initial_screenshot: bool = False
    ) -> list[Any]:
        """
        Create initial messages for the conversation.

        Args:
            prompt: The user's prompt
            initial_screenshot: Whether to capture initial screenshot

        Returns:
            List of messages in provider-specific format
        """

    @abstractmethod
    async def get_model_response(self, messages: list[Any]) -> AgentResponse:
        """
        Get response from the model including any tool calls.

        NOTE: Subclasses should decorate this method with:
            @hud.instrument(span_type="agent", record_args=False, record_result=True)

        Args:
            messages: Current conversation messages

        Returns:
            AgentResponse with content, tool_calls, and done fields
        """

    @abstractmethod
    async def format_tool_results(
        self, tool_calls: list[MCPToolCall], tool_results: list[MCPToolResult]
    ) -> list[Any]:
        """
        Format tool results into messages for the model.

        Args:
            tool_calls: List of MCPToolCall objects that were executed
            tool_results: List of MCPToolResult objects from tool execution

        Returns:
            List of formatted messages to append to conversation
        """
        raise NotImplementedError

    async def create_user_message(self, text: str) -> Any:
        """
        Create a user message in the format expected by the model.

        Default implementation for text-only messages.
        Subclasses can override for specific formats.

        Args:
            text: User's text input

        Returns:
            Formatted user message
        """
        return {"role": "user", "content": text}


def _find_reward(result: MCPToolResult) -> float:
    """Find the reward in the result.

    Agent accepts "reward", "grade", "score"

    If not found, return 0.0
    """
    accept_keys = ["reward", "grade", "score"]
    for key in accept_keys:
        if isinstance(result.structuredContent, dict) and key in result.structuredContent:
            return result.structuredContent[key]
    if isinstance(result.content, list):
        for content in result.content:
            if isinstance(content, types.TextContent):
                try:
                    json_content = json.loads(content.text)
                    for key, value in json_content.items():
                        if key in accept_keys:
                            return value
                except json.JSONDecodeError:
                    pass
    return 0.0


def _find_content(result: MCPToolResult) -> str | None:
    """Find the content in the result.

    Agent accepts "content", "text", "message"

    If not found, return 0.0
    """
    accept_keys = ["content", "logs"]
    for key in accept_keys:
        if isinstance(result.structuredContent, dict) and key in result.structuredContent:
            return result.structuredContent[key]
    return None
