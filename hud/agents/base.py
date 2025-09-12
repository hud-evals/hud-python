"""Base MCP Agent implementation."""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import mcp.types as types

from hud.types import AgentResponse, MCPToolCall, MCPToolResult, Trace
from hud.utils.hud_console import HUDConsole
from hud.utils.mcp import MCPConfigPatch, patch_mcp_config, setup_hud_telemetry

if TYPE_CHECKING:
    from hud.clients.base import AgentMCPClient
    from hud.datasets import Task

    from .misc import ResponseAgent


logger = logging.getLogger(__name__)

GLOBAL_SYSTEM_PROMPT = "You are an assistant that can use tools to help the user. You will be given a task and you will need to use the tools to complete the task."  # noqa: E501


class MCPAgent(ABC):
    """
    Base class for MCP-enabled agents.

    Provides common behavior for agents that interact with MCP servers, including:
    - Client management: accepts an `AgentMCPClient` or auto-creates one at
      runtime when `run()` is called with a `Task` that includes `mcp_config`.
    - Tool lifecycle: discovery, filtering (`allowed_tools`, `disallowed_tools`),
      and automatic marking of lifecycle tools (setup/evaluate) from a `Task`.
    - Messaging: system prompt handling, optional inclusion of setup output on
      the first turn, and control over initial screenshots.
    - Telemetry & UX: standardized logging/printing via `HUDConsole` and optional
      automatic tracing (`auto_trace`).

    Subclasses implement provider-specific formatting and response fetching
    by overriding these abstract methods: `get_system_messages`, `get_response`,
    `format_blocks`, and `format_tool_results`.
    """

    metadata: dict[str, Any]
    required_tools: ClassVar[list[str]] = []  # Tools that must be available

    def __init__(
        self,
        mcp_client: AgentMCPClient | None = None,
        # Filtering
        allowed_tools: list[str] | None = None,
        disallowed_tools: list[str] | None = None,
        lifecycle_tools: list[str] | None = None,
        # Messages
        system_prompt: str = GLOBAL_SYSTEM_PROMPT,
        append_setup_output: bool = True,
        initial_screenshot: bool = True,
        # Misc
        model_name: str = "mcp-agent",
        response_agent: ResponseAgent | None = None,
        auto_trace: bool = True,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the base MCP agent.

        Args:
            mcp_client: Client for connecting to MCP servers. If None, a client
                is auto-created at runtime when `run()` is called with a `Task`
                that provides `mcp_config`.
            allowed_tools: Names of tools to allow (None means allow all).
            disallowed_tools: Names of tools to always exclude.
            lifecycle_tools: Tools reserved for lifecycle phases (e.g., setup,
                evaluate). These are hidden from normal tool calling.
            system_prompt: System prompt to seed the conversation.
            append_setup_output: Whether to append setup tool output to the
                first turn's messages.
            initial_screenshot: Whether to include an initial screenshot before
                the first prompt (when supported by the environment).
            model_name: Label used in telemetry/logging to identify the model.
            response_agent: Optional automation that can respond to the model's
                outputs to keep the loop going (e.g., auto-continue/stop).
            auto_trace: If True, automatically creates a trace/span for runs.
            verbose: If True, increases logging verbosity for developer UX.
        """

        self.mcp_client = mcp_client
        self._auto_created_client = False  # Track if we created the client

        self.model_name = model_name
        self.console = HUDConsole(logger=logger)

        # Set verbose mode if requested
        if verbose:
            self.console.set_verbose(True)

        # Filtering
        self.allowed_tools = allowed_tools
        self.disallowed_tools = disallowed_tools or []
        self.lifecycle_tools = lifecycle_tools or []

        # Messages
        self.system_prompt = system_prompt
        self.append_setup_output = append_setup_output
        self.initial_screenshot = initial_screenshot

        # Initialize these here so methods can be called before initialize()
        self._available_tools: list[types.Tool] = []
        self._tool_map: dict[str, types.Tool] = {}  # Simplified: just name to tool
        self.response_tool_name = None
        self.initialization_complete = False

        # Trace
        self._auto_trace = auto_trace
        self._auto_trace_cm: Any | None = None  # Store auto-created trace context manager

        # Response agent to automatically interact with the model
        self.response_agent = response_agent

    async def initialize(self, task: str | Task | None = None) -> None:
        """Initialize the agent with task-specific configuration."""
        from hud.datasets import Task

        # Create client if needed
        if self.mcp_client is None and isinstance(task, Task) and task.mcp_config:
            from hud.clients import MCPClient

            self.mcp_client = MCPClient(mcp_config=task.mcp_config)
            self._auto_created_client = True
            self.console.info_log("Auto-created MCPClient from task.mcp_config")

        # Ensure we have a client
        if self.mcp_client is None:
            raise ValueError(
                "No MCPClient. Please provide one when initializing the agent or pass a Task with mcp_config."  # noqa: E501
            )

        await self._setup_config(self.mcp_client.mcp_config)

        # Initialize client if needed
        try:
            await self.mcp_client.initialize()
        except Exception as e:
            self._handle_connection_error(e)

        # If task is provided, add lifecycle tools
        if isinstance(task, Task):
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
            if task.system_prompt:
                self.system_prompt += "\n\n" + task.system_prompt

        # Re-apply filtering with updated lifecycle tools
        await self._filter_tools()

        num_tools = len(self._available_tools)
        self.console.success_log(
            f"Agent initialized with {num_tools} available tools (after filtering)"
        )

    async def run(self, prompt_or_task: str | Task | dict[str, Any], max_steps: int = 10) -> Trace:
        """
        Run the agent with the given prompt or task.

        Args:
            prompt_or_task: Either a string prompt for simple execution or a Task object
            max_steps: Maximum number of steps (-1 for infinite)

        Returns:
            Trace with reward, done, content, isError fields and trace steps
        """
        # Import here to avoid circular imports
        from hud.datasets import Task

        if isinstance(prompt_or_task, dict):
            prompt_or_task = Task(**prompt_or_task)

        try:
            # Establish the connection with the MCP server/Environment
            if not self.initialization_complete:
                await self.initialize(prompt_or_task)
                self.initialization_complete = True

            # Handle Task objects with full lifecycle
            if isinstance(prompt_or_task, Task):
                return await self.run_task(prompt_or_task, max_steps)

            # Handle simple string prompts
            elif isinstance(prompt_or_task, str):
                context = text_to_blocks(prompt_or_task)
                return await self._run_context(context, max_steps=max_steps)

            else:
                raise TypeError(f"prompt_or_task must be str or Task, got {type(prompt_or_task)}")
        except Exception as e:
            # Always return a Trace object for any exception
            if self._is_connection_error(e):
                # Return error trace for connection failures
                return Trace(
                    reward=0.0,
                    done=True,
                    content=self._get_connection_error_message(e),
                    isError=True,
                )
            else:
                # Return error trace for any other exception
                return Trace(
                    reward=0.0,
                    done=True,
                    content=f"Task failed with error: {e}",
                    isError=True,
                    info={"error": str(e)},
                )
        finally:
            # Cleanup auto-created resources
            await self._cleanup()

    async def run_task(self, task: Task, max_steps: int = 10) -> Trace:
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
            start_context: list[types.ContentBlock] = []

            # Extract the initial task information
            if task.prompt:
                start_context.extend(text_to_blocks(task.prompt))

            # Execute the setup tool and append the initial observation to the context
            if task.setup_tool is not None:
                self.console.progress_log(f"Setting up tool phase: {task.setup_tool}")
                results = await self.call_tools(task.setup_tool)
                if any(result.isError for result in results):
                    raise RuntimeError(f"{results}")

                if self.append_setup_output and isinstance(results[0].content, list):
                    start_context.extend(results[0].content)
            if not self.initial_screenshot:
                start_context = await self._filter_messages(start_context, include_types=["text"])

            # Execute the task (agent loop) - this returns a empty trace object with the final response  # noqa: E501
            prompt_result = await self._run_context(start_context, max_steps=max_steps)

        except Exception as e:
            self.console.error_log(f"Task execution failed: {e}")
            # Create an error result but don't return yet - we still want to evaluate
            prompt_result = Trace(reward=0.0, done=True, content=str(e), isError=True)
            prompt_result.populate_from_context()

        # Always evaluate if we have evaluate tool, regardless of errors
        if task.evaluate_tool is not None:
            try:
                self.console.progress_log(f"Evaluating tool phase: {task.evaluate_tool}")
                results = await self.call_tools(task.evaluate_tool)

                if any(result.isError for result in results):
                    self.console.warning_log(f"Evaluate tool returned error: {results}")
                    # Still extract what we can from the error response
                    if prompt_result is None:
                        prompt_result = Trace(
                            reward=0.0,
                            done=True,
                            content="Task failed before evaluation",
                            isError=True,
                        )
                    prompt_result.reward = 0.0  # Default to 0 on error
                else:
                    # Extract reward and content from evaluation
                    if results:
                        reward = find_reward(results[0])
                        eval_content = find_content(results[0])

                        # Update the prompt result with evaluation reward
                        if prompt_result is None:
                            prompt_result = Trace(
                                reward=reward, done=True, content=eval_content or "", isError=False
                            )
                        else:
                            prompt_result.reward = reward

                            # Update the prompt result with evaluation content (if available)
                            if eval_content:
                                # Prompt result may already have final response content,
                                # so we append to it
                                if prompt_result.content:
                                    prompt_result.content += "\n\n" + eval_content
                                else:
                                    prompt_result.content = eval_content

            except Exception as e:
                self.console.error_log(f"Evaluation phase failed: {e}")
                # Ensure we have a result even if evaluation failed
                if prompt_result is None:
                    prompt_result = Trace(
                        reward=0.0, done=True, content=f"Evaluation failed: {e}", isError=True
                    )

        return (
            prompt_result
            if prompt_result
            else Trace(reward=0.0, done=True, content="No result available", isError=True)
        )

    async def _run_context(
        self, context: list[types.ContentBlock], *, max_steps: int = 10
    ) -> Trace:
        """
        Run the agent with the given context messages. This is the core agent loop.

        Args:
            context: The context to complete
            max_steps: Maximum number of steps (-1 for infinite)

        Returns:
            Trace with reward, done, content fields and trace steps
        """
        final_response = None
        error = None

        try:
            # Start with system messages
            messages = await self.get_system_messages()

            # Add initial context
            messages.extend(await self.format_message(context))
            self.console.debug(f"Messages: {messages}")

            step_count = 0
            while max_steps == -1 or step_count < max_steps:
                step_count += 1
                if max_steps == -1:
                    self.console.debug(f"Step {step_count} (unlimited)")
                else:
                    self.console.debug(f"Step {step_count}/{max_steps}")

                try:
                    # 1. Get model response
                    response = await self.get_response(messages)

                    self.console.debug(f"Agent:\n{response}")

                    # Check if we should stop
                    if response.done or not response.tool_calls:
                        # Optional external ResponseAgent to decide whether to stop
                        decision = "STOP"
                        if self.response_agent is not None and response.content:
                            try:
                                decision = await self.response_agent.determine_response(
                                    response.content
                                )
                            except Exception as e:
                                self.console.warning_log(f"ResponseAgent failed: {e}")
                        if decision == "STOP":
                            # Try to submit response through lifecycle tool
                            await self._maybe_submit_response(response, messages)

                            self.console.debug("Stopping execution")
                            final_response = response
                            break
                        else:
                            self.console.debug("Continuing execution")
                            messages.extend(await self.format_message(decision))
                            continue

                    # 2. Execute tools
                    tool_calls = response.tool_calls
                    tool_results = await self.call_tools(tool_calls)

                    # 3. Format tool results and add to messages
                    tool_messages = await self.format_tool_results(tool_calls, tool_results)
                    messages.extend(tool_messages)

                    # Compact step completion display
                    step_info = f"\n[bold]Step {step_count}"
                    if max_steps != -1:
                        step_info += f"/{max_steps}"
                    step_info += "[/bold]"

                    # Show tool calls and results in compact format
                    for call, result in zip(tool_calls, tool_results, strict=False):
                        step_info += f"\n{call}\n{result}"

                    self.console.info_log(step_info)

                except Exception as e:
                    self.console.error_log(f"Step failed: {e}")
                    error = str(e)
                    break

        except KeyboardInterrupt:
            self.console.warning_log("Agent execution interrupted by user")
            error = "Interrupted by user"
        except asyncio.CancelledError:
            self.console.warning_log("Agent execution cancelled")
            error = "Cancelled"
        except Exception as e:
            self.console.error_log(f"Unexpected error: {e}")
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

    async def call_tools(
        self, tool_call: MCPToolCall | list[MCPToolCall] | None = None
    ) -> list[MCPToolResult]:
        """
        Call a tool through the MCP client.

        Args:
            tool_call: MCPToolCall or list of MCPToolCall

        Returns:
            List of MCPToolResult
        """
        if tool_call is None:
            return []

        if isinstance(tool_call, MCPToolCall):
            tool_call = [tool_call]

        if self.mcp_client is None:
            raise ValueError("Client is not initialized")

        results: list[MCPToolResult] = []
        for tc in tool_call:
            try:
                self.console.debug(f"Calling tool: {tc}")
                results.append(await self.mcp_client.call_tool(tc))
            except TimeoutError as e:
                self.console.error_log(f"Tool execution timed out: {e}")
                try:
                    await self.mcp_client.shutdown()
                except Exception as close_err:
                    self.console.debug(f"Failed to close MCP client cleanly: {close_err}")
                raise
            except Exception as e:
                self.console.error_log(f"Tool execution failed: {e}")
                results.append(_format_error_result(str(e)))
        return results

    @abstractmethod
    async def get_system_messages(self) -> list[Any]:
        """
        Get the system prompt.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_response(
        self, messages: list[Any]
    ) -> AgentResponse:  # maybe type messages as list[types.ContentBlock]
        """
        Get response from the model including any tool calls.

        NOTE: Subclasses should decorate this method with:
            @hud.instrument(span_type="agent", record_args=False, record_result=True)

        Args:
            messages: Current conversation messages

        Returns:
            AgentResponse with content, tool_calls, and done fields
        """
        raise NotImplementedError

    @abstractmethod
    async def format_blocks(self, blocks: list[types.ContentBlock]) -> list[Any]:
        """
        Format a list of content blocks into a list of messages.
        """
        raise NotImplementedError

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

    async def format_message(
        self,
        message: str
        | list[str]
        | types.ContentBlock
        | list[types.ContentBlock]
        | list[str | types.ContentBlock],
    ) -> list[Any]:  # maybe type messages as list[types.ContentBlock]
        """
        Convencience function.

        Format a single content message into a list of messages for the model.
        """
        blocks: list[types.ContentBlock] = []
        if not isinstance(message, list):
            message = [message]

        for m in message:
            if isinstance(m, str):
                blocks.append(types.TextContent(text=m, type="text"))
            elif isinstance(m, types.ContentBlock):
                blocks.append(m)
            else:
                raise ValueError(f"Invalid message type: {type(m)}")

        return await self.format_blocks(blocks)

    async def _filter_tools(self) -> None:
        """Apply tool filtering based on allowed/disallowed lists."""
        # Get all tools from client
        if self.mcp_client is None:
            raise ValueError("MCP client is not initialized")

        all_tools = await self.mcp_client.list_tools()

        # Filter tools
        self._available_tools = []
        self._tool_map = {}

        # Track response tools by server
        response_tools_by_server: dict[str, str] = {}  # server_name -> tool_name

        for tool in all_tools:
            # Check if tool should be included
            if self.allowed_tools and tool.name not in self.allowed_tools:
                continue
            if tool.name in self.disallowed_tools:
                continue

            self._available_tools.append(tool)
            # Simplified mapping - just tool name to tool
            self._tool_map[tool.name] = tool

            # Track response tools
            if "response" in tool.name or tool.name == "response":
                # Extract server name from tool name (e.g., "grader_response" -> "grader")
                if "_" in tool.name:
                    server_name = tool.name.split("_", 1)[0]
                    response_tools_by_server[server_name] = tool.name
                else:
                    response_tools_by_server["_default"] = tool.name

        # Find the response tool to use (prioritize last server in config)
        if response_tools_by_server and hasattr(self.mcp_client, "mcp_config"):
            # Get server names in order from mcp_config
            server_names = list(self.mcp_client.mcp_config.keys())

            # Try to find response tool from last server first
            response_tool_name = None
            for server_name in reversed(server_names):
                if server_name in response_tools_by_server:
                    response_tool_name = response_tools_by_server[server_name]
                    break

            # Fallback to any response tool
            if not response_tool_name and response_tools_by_server:
                response_tool_name = next(iter(response_tools_by_server.values()))

            # Add to lifecycle tools if found
            if response_tool_name and response_tool_name not in self.lifecycle_tools:
                self.console.debug(f"Auto-detected '{response_tool_name}' tool as a lifecycle tool")
                self.response_tool_name = response_tool_name
                self.lifecycle_tools.append(response_tool_name)

        # Check if all required tools are available
        if self.required_tools:
            available_tool_names = {tool.name for tool in self._available_tools}
            missing_tools = [
                tool for tool in self.required_tools if tool not in available_tool_names
            ]
            if missing_tools:
                raise ValueError(
                    f"Required tools not available: {missing_tools}. "
                    f"Available tools: {list(available_tool_names)}"
                )

    async def _maybe_submit_response(self, response: AgentResponse, messages: list[Any]) -> None:
        """Submit response through lifecycle tool if available.

        Args:
            response: The agent's response
            messages: The current message history (will be modified in-place)
        """
        if self.response_tool_name:
            self.console.debug(f"Calling response lifecycle tool: {self.response_tool_name}")
            try:
                # Call the response tool with the agent's response
                response_tool_call = MCPToolCall(
                    name=self.response_tool_name, arguments={"response": response.content}
                )
                response_results = await self.call_tools(response_tool_call)

                # Format and add the response tool results to messages
                response_messages = await self.format_tool_results(
                    [response_tool_call], response_results
                )
                messages.extend(response_messages)

                # Mark the task as done
                self.console.debug("Response lifecycle tool executed, marking task as done")
            except Exception as e:
                self.console.error_log(f"Response lifecycle tool failed: {e}")

    async def _setup_config(self, mcp_config: dict[str, dict[str, Any]]) -> None:
        """Inject metadata into the metadata of the initialize request."""
        if self.metadata:
            patch_mcp_config(
                mcp_config,
                MCPConfigPatch(meta=self.metadata),
            )
        self._auto_trace_cm = setup_hud_telemetry(mcp_config, auto_trace=self._auto_trace)

    def get_available_tools(self) -> list[types.Tool]:
        """Get list of available MCP tools for LLM use (excludes lifecycle tools)."""
        lifecycle_tool_names = self.lifecycle_tools
        return [tool for tool in self._available_tools if tool.name not in lifecycle_tool_names]

    def get_tool_schemas(self) -> list[dict]:
        """Get tool schemas in a format suitable for the model."""
        schemas = []
        for tool in self.get_available_tools():
            schema = {
                "name": tool.name,
                "description": tool.description,
            }
            if tool.inputSchema:
                schema["parameters"] = tool.inputSchema
            schemas.append(schema)
        return schemas

    async def _filter_messages(
        self,
        message_list: list[types.ContentBlock],
        include_types: list[
            Literal["text", "image", "audio", "resource_link", "embedded_resource"]
        ],
    ) -> list[types.ContentBlock]:
        """
        Filter a list of messages and return only the messages of the given types.

        Args:
            message_list: The list of messages to filter
            include_types: List of types to include (None = all types)

        Returns:
            List of messages in provider-specific format
        """
        return [message for message in message_list if message.type in include_types]

    async def _cleanup(self) -> None:
        """Cleanup resources."""
        # Clean up auto-created trace if any
        if self._auto_trace_cm:
            try:
                self._auto_trace_cm.__exit__(None, None, None)
                self.console.debug("Closed auto-created trace")
            except Exception as e:
                self.console.warning_log(f"Failed to close auto-created trace: {e}")
            finally:
                self._auto_trace_cm = None

        # Clean up auto-created client
        if self._auto_created_client and self.mcp_client:
            try:
                await self.mcp_client.shutdown()
                self.console.debug("Closed auto-created MCPClient")
            except Exception as e:
                self.console.warning_log(f"Failed to close auto-created client: {e}")
            finally:
                self.mcp_client = None
                self._auto_created_client = False

    def _is_connection_error(self, e: Exception) -> bool:
        """Check if an exception is a connection error."""
        error_msg = str(e).lower()
        return any(
            pattern in error_msg
            for pattern in [
                "connection",
                "connect",
                "refused",
                "failed",
                "could not connect",
                "mcp server",
            ]
        )

    def _get_connection_error_message(self, e: Exception) -> str:
        """Extract a helpful connection error message."""
        import re

        url_match = re.search(r"https?://[^\s]+", str(e))
        url = url_match.group(0) if url_match else "the MCP server"
        return f"Connection failed: Could not connect to {url}. Is your MCP client/server running?"

    def _handle_connection_error(self, e: Exception) -> None:
        """Handle connection errors with helpful messages."""
        if self._is_connection_error(e):
            msg = self._get_connection_error_message(e)
            # Always show connection errors, not just when logging is enabled
            self.console.error(f"❌ {msg}")
            self.console.info("💡 Make sure the MCP server is started before running the agent.")

            # For localhost, provide specific instructions
            error_str = str(e).lower()
            if "localhost" in error_str or "127.0.0.1" in error_str:
                self.console.info("   Run 'hud dev' in another terminal to start the MCP server")

            raise RuntimeError(msg) from e
        raise


def _format_error_result(error_message: str) -> MCPToolResult:
    return MCPToolResult(content=text_to_blocks(error_message), isError=True)


def text_to_blocks(text: str) -> list[types.ContentBlock]:
    return [types.TextContent(text=text, type="text")]


def find_reward(result: MCPToolResult) -> float:
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


def find_content(result: MCPToolResult) -> str | None:
    """Find the content in the result.

    Agent accepts "content", "text", "message", or "logs"

    If not found, return 0.0
    """
    accept_keys = ["content", "text", "message", "logs"]
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
    return ""
