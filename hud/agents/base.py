"""Base MCP Agent implementation."""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import mcp.types as types
from pydantic import BaseModel, ConfigDict

from hud.types import AgentResponse, BaseAgentConfig, MCPToolCall, MCPToolResult, Trace
from hud.utils.hud_console import HUDConsole

if TYPE_CHECKING:
    from hud.environment import Environment
    from hud.eval.context import EvalContext


logger = logging.getLogger(__name__)


class BaseCreateParams(BaseModel):
    """Runtime parameters for agent creation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Primary way to bind agent to execution context (v5)
    ctx: Any | None = None  # EvalContext or Environment - agent uses this for tool calls

    auto_respond: bool = False
    verbose: bool = False


class MCPAgent(ABC):
    """
    Base class for MCP-enabled agents.

    Agents interact with MCP servers through an EvalContext:
    - run(ctx): Main entry point - takes EvalContext from hud.eval()
    - ctx.call_tool(): Used internally for all tool execution
    - ctx.submit(): Called automatically with agent's final response

    Subclasses implement provider-specific formatting and response fetching
    by overriding: `get_system_messages`, `get_response`, `format_blocks`,
    and `format_tool_results`.
    """

    metadata: ClassVar[dict[str, Any] | None] = None
    required_tools: ClassVar[list[str]] = []  # Tools that must be available
    config_cls: ClassVar[type[BaseAgentConfig]] = BaseAgentConfig

    def __init__(self, params: BaseCreateParams | None = None, **kwargs: Any) -> None:
        if params is None:
            import warnings

            warnings.warn(
                f"Passing kwargs to {self.__class__.__name__}() is deprecated. "
                f"Use {self.__class__.__name__}.create(...) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            CreateParams = type(
                f"{self.config_cls.__name__}CreateParams",
                (BaseCreateParams, self.config_cls),
                {"__module__": self.config_cls.__module__},
            )
            params = CreateParams(**kwargs)

        config_kwargs = {
            k: getattr(params, k) for k in self.config_cls.model_fields if hasattr(params, k)
        }
        self.config = self.config_cls(**config_kwargs)

        # v5: Store execution context (EvalContext/Environment) - agent uses ctx.call_tool()
        self.ctx: EvalContext | Environment | None = params.ctx

        self.model_name: str = getattr(params, "model_name", "MCPAgent")
        self.model: str = getattr(params, "model", None) or "unknown"
        self.auto_respond = params.auto_respond

        self.console = HUDConsole(logger=logger)

        if params.verbose:
            self.console.set_verbose(True)

        self.system_prompt = self.config.system_prompt

        self._available_tools: list[types.Tool] | None = None
        self._tool_map: dict[str, types.Tool] = {}
        self._initialized: bool = False

    @classmethod
    def create(cls, **kwargs: Any) -> MCPAgent:
        """
        Factory method to create an agent with typed parameters.
        """
        CreateParams = type(
            f"{cls.config_cls.__name__}CreateParams",
            (BaseCreateParams, cls.config_cls),
            {"__module__": cls.config_cls.__module__},
        )
        return cls(params=CreateParams(**kwargs))

    async def _initialize_from_ctx(self, ctx: EvalContext) -> None:
        """Initialize agent from EvalContext - discovers tools and sets up state.

        This is the v5 initialization path. The agent uses ctx.call_tool() directly
        for tool execution (no EnvironmentClient wrapper needed).
        """
        from hud.eval.context import EvalContext

        if not isinstance(ctx, EvalContext):
            raise TypeError(f"ctx must be EvalContext, got {type(ctx).__name__}")

        # Refresh tools from connections, then get filtered list for agent
        await ctx.list_tools()
        self._available_tools = ctx.as_tools()
        self._tool_map = {t.name: t for t in self._available_tools}

        # Validate required tools are present
        available_tool_names = {t.name for t in self._available_tools}
        missing_tools = [tool for tool in self.required_tools if tool not in available_tool_names]
        if missing_tools:
            raise ValueError(
                f"Required tools are missing: {missing_tools}. "
                f"Available tools: {sorted(available_tool_names)}"
            )

        self.console.info(
            f"Agent initialized with {len(self._available_tools)} tools: "
            f"{', '.join([t.name for t in self._available_tools])}"
        )

        # Call hook for subclass-specific initialization (e.g., tool format conversion)
        self._on_tools_ready()

        self._initialized = True

    def _on_tools_ready(self) -> None:
        """Hook called after tools are discovered and validated.

        Subclasses can override this to perform provider-specific setup,
        such as converting MCP tools to the provider's format.

        Called by _initialize_from_ctx() after _available_tools is populated.
        """
        return  # Default no-op - subclasses override for provider-specific setup

    async def run(
        self,
        ctx: EvalContext,
        *,
        max_steps: int = 10,
    ) -> Trace:
        """
        Run the agent on the given evaluation context.

        The agent uses ctx.prompt as the task and ctx.call_tool() for tool execution.
        Automatically calls ctx.submit() with the final answer.

        Args:
            ctx: EvalContext from hud.eval() - contains prompt and tools
            max_steps: Maximum number of agent steps (-1 for infinite)

        Returns:
            Trace with done, content, isError fields

        Example:
            ```python
            async with hud.eval(task) as ctx:
                agent = ClaudeAgent.create()
                await agent.run(ctx)
            # ctx.reward is set by the scenario's evaluate phase
            ```
        """
        from hud.eval.context import EvalContext

        if not isinstance(ctx, EvalContext):
            raise TypeError(f"ctx must be EvalContext, got {type(ctx).__name__}")

        if not ctx.prompt:
            raise ValueError("ctx.prompt is not set - did the scenario setup run?")

        # Store context for tool calls
        self.ctx = ctx

        # Initialize tools from context
        if not self._initialized:
            await self._initialize_from_ctx(ctx)

        try:
            result = await self._run_context(text_to_blocks(ctx.prompt), max_steps=max_steps)

            # Submit final answer to context (only if scenario is running)
            if result.content and ctx.has_scenario:
                await ctx.submit(result.content)

            return result

        except Exception as e:
            logger.exception("Error while running agent:")
            return Trace(
                reward=0.0,
                done=True,
                content=f"Agent failed with error: {e}",
                isError=True,
                info={"error": str(e)},
            )
        finally:
            # Cleanup auto-created resources
            await self._cleanup()

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

        messages: list[Any] = []

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
                        # Use auto_respond to decide whether to stop
                        decision: Literal["STOP", "CONTINUE"] = "STOP"
                        if self.auto_respond and response.content:
                            try:
                                from hud.agents.misc import ResponseAgent

                                response_agent = ResponseAgent()
                                decision = await response_agent.determine_response(response.content)
                            except Exception as e:
                                self.console.warning_log(f"Auto-respond failed: {e}")
                        if decision == "STOP":
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
        if error is not None or (
            final_response and hasattr(final_response, "isError") and final_response.isError
        ):
            is_error = True
        else:
            is_error = False

        # Ensure all parameters are the correct type
        trace_params = {
            "reward": 0.0,
            "done": True,
            "messages": messages,
            "content": final_response.content if final_response else error,
            "isError": is_error,
            "info": {"error": error} if error else {},
        }
        trace_result = Trace(**trace_params)

        return trace_result

    async def call_tools(
        self, tool_call: MCPToolCall | list[MCPToolCall] | None = None
    ) -> list[MCPToolResult]:
        """
        Call tools through the bound EvalContext.

        Args:
            tool_call: MCPToolCall or list of MCPToolCall

        Returns:
            List of MCPToolResult
        """
        if tool_call is None:
            return []

        if isinstance(tool_call, MCPToolCall):
            tool_call = [tool_call]

        if self.ctx is None:
            raise ValueError("Agent not bound to context - call run(ctx) first")

        results: list[MCPToolResult] = []
        for tc in tool_call:
            try:
                self.console.debug(f"Calling tool: {tc}")
                result = await self.ctx.call_tool(tc)
                results.append(MCPToolResult(content=result.content, isError=result.isError))
            except TimeoutError as e:
                self.console.error_log(f"Tool execution timed out: {e}")
                raise
            except Exception as e:
                self.console.error_log(f"Tool execution failed: {e}")
                results.append(_format_error_result(str(e)))
        return results

    @abstractmethod
    async def get_system_messages(self) -> list[types.ContentBlock]:
        """
        Get the system prompt.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_response(self, messages: list[Any]) -> AgentResponse:
        """
        Get response from the model including any tool calls.


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

    def get_available_tools(self) -> list[types.Tool]:
        """Get list of available MCP tools for LLM use (excludes lifecycle tools)."""
        if self._available_tools is None:
            raise RuntimeError(
                "Tools have not been initialized. Call initialize() before accessing available tools."  # noqa: E501
            )
        return self._available_tools

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
        # Clear context reference
        self.ctx = None


def _format_error_result(error_message: str) -> MCPToolResult:
    return MCPToolResult(content=text_to_blocks(error_message), isError=True)


def text_to_blocks(text: str) -> list[types.ContentBlock]:
    return [types.TextContent(text=text, type="text")]


def find_reward(result: MCPToolResult) -> float:
    """Find the reward in the result.

    Agent accepts "reward", "grade", "score", or weighted subscores

    If not found, return 0.0
    """
    accept_keys = ["reward", "grade", "score"]

    # Check for direct reward/grade/score keys
    for key in accept_keys:
        if isinstance(result.structuredContent, dict) and key in result.structuredContent:
            return result.structuredContent[key]

    # Check for subscores and weights format
    if (
        isinstance(result.structuredContent, dict)
        and "subscores" in result.structuredContent
        and "weights" in result.structuredContent
    ):
        subscores = result.structuredContent["subscores"]
        weights = result.structuredContent["weights"]
        if isinstance(subscores, dict) and isinstance(weights, dict):
            try:
                # Multiply each subscore by its corresponding weight and sum
                reward = sum(
                    float(subscores[key]) * float(weights.get(key, 0.0))
                    for key in subscores
                    if key in weights
                )
                return reward
            except (ValueError, TypeError) as e:
                logger.error("Failed to parse subscores/weights: %s", e)
                return 0.0

    # Check for reward in JSON text content
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

    logger.error("Couldn't parse reward from result: %s", result)
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
