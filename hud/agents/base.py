"""Base MCP Agent implementation."""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import mcp.types as types

from hud.tools.native_types import NativeToolSpec
from hud.types import AgentResponse, AgentType, BaseAgentConfig, MCPToolCall, MCPToolResult, Trace
from hud.utils.hud_console import HUDConsole

from .types import BaseCreateParams

if TYPE_CHECKING:
    from hud.environment import Environment
    from hud.eval.context import EvalContext


logger = logging.getLogger(__name__)


@dataclass
class CategorizedTools:
    """Result of categorizing tools by native spec availability.

    Used by agents to efficiently process tools with shared logic for
    role-based mutual exclusion.
    """

    native: list[tuple[types.Tool, NativeToolSpec]] = field(default_factory=list)
    """Tools with native specs for this agent (tool, spec) pairs."""

    hosted: list[tuple[types.Tool, NativeToolSpec]] = field(default_factory=list)
    """Hosted tools with native specs for this agent (tool, spec) pairs."""

    generic: list[types.Tool] = field(default_factory=list)
    """Tools without native specs that aren't role-blocked."""

    claimed_roles: set[str] = field(default_factory=set)
    """Roles claimed by native tools."""

    skipped: list[tuple[types.Tool, str]] = field(default_factory=list)
    """Tools skipped due to role conflicts (tool, reason) pairs."""


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

    @classmethod
    @abstractmethod
    def agent_type(cls) -> AgentType:
        """Return the AgentType for this agent.

        Subclasses must implement this to return their corresponding AgentType enum value.
        This is used for resolving native tool specifications.

        Returns:
            AgentType enum value for this agent
        """
        raise NotImplementedError

    def resolve_native_spec(self, tool: types.Tool) -> NativeToolSpec | None:
        """Check if a tool has a native spec for this agent type and model.

        Looks up the tool's meta.native_tools field for a spec matching this agent's type.
        If found, validates that the current model supports this native spec.
        Returns a NativeToolSpec that can be used to register the tool with
        the provider's native API format.

        Falls back to legacy name-based detection for backwards compatibility with
        old environments that don't emit native_tools metadata.

        Args:
            tool: MCP Tool object to check for native specs

        Returns:
            NativeToolSpec if the tool has a native spec for this agent and the
            current model supports it, None otherwise. When the model doesn't
            match supported_models, returns None so the tool falls back to
            generic function calling.
        """
        spec: NativeToolSpec | None = None

        # First try metadata-based resolution
        if tool.meta:
            native_tools = tool.meta.get("native_tools", {})
            spec_dict = native_tools.get(self.agent_type().value)

            if spec_dict and isinstance(spec_dict, dict):
                # Extract known fields and put the rest in extra
                known_fields = {
                    "api_type",
                    "api_name",
                    "beta",
                    "hosted",
                    "role",
                    "supported_models",
                }
                extra = {k: v for k, v in spec_dict.items() if k not in known_fields}

                # Convert supported_models list to tuple for frozen model
                supported_models_raw = spec_dict.get("supported_models")
                supported_models: tuple[str, ...] | None = None
                if supported_models_raw:
                    supported_models = tuple(supported_models_raw)

                spec = NativeToolSpec(
                    api_type=spec_dict.get("api_type"),
                    api_name=spec_dict.get("api_name"),
                    beta=spec_dict.get("beta"),
                    hosted=spec_dict.get("hosted", False),
                    role=spec_dict.get("role"),
                    supported_models=supported_models,
                    extra=extra,
                )

        # Fall back to legacy name-based detection for old environments
        if spec is None:
            spec = self._legacy_native_spec_fallback(tool)

        # Check if current model supports this native spec
        if spec is not None and not spec.supports_model(self.model):
            logger.debug(
                "Model %s not in supported_models for native spec %s, falling back to functions",
                self.model,
                spec.api_type,
            )
            return None

        return spec

    def _legacy_native_spec_fallback(self, tool: types.Tool) -> NativeToolSpec | None:
        """Detect native tools by name for backwards compatibility.

        Override in subclasses to support old environments that expose tools
        without native_tools metadata.

        Args:
            tool: MCP Tool object to check

        Returns:
            NativeToolSpec if the tool matches a known legacy pattern, None otherwise
        """
        return None

    def get_tool_role(self, tool: types.Tool) -> str | None:
        """Get the role of a tool from any of its native specs.

        The role is used for mutual exclusion - when an agent accepts a tool
        natively, other tools with the same role are excluded.

        Args:
            tool: MCP Tool object to check

        Returns:
            The role string if any native spec defines one, None otherwise
        """
        if not tool.meta:
            return None

        native_tools = tool.meta.get("native_tools", {})
        if not native_tools:
            return None

        # Check all specs for a role (they should all have the same role)
        for spec_dict in native_tools.values():
            if isinstance(spec_dict, dict) and spec_dict.get("role"):
                return spec_dict["role"]

        return None

    def categorize_tools(self, tools: list[types.Tool] | None = None) -> CategorizedTools:
        """Categorize tools by native spec availability with role-based exclusion.

        This shared method implements the two-pass tool processing logic:
        1. First pass: identify native/hosted tools and claim their roles
        2. Second pass: include generic tools if their role isn't claimed

        Args:
            tools: List of MCP tools to categorize. If None, uses get_available_tools()

        Returns:
            CategorizedTools with native, hosted, generic, and skipped tools
        """
        if tools is None:
            tools = self.get_available_tools()

        result = CategorizedTools()

        # First pass: process tools with native specs for this agent
        for tool in tools:
            spec = self.resolve_native_spec(tool)
            if not spec:
                continue

            # Check for role conflicts between native tools
            if spec.role:
                if spec.role in result.claimed_roles:
                    # Another native tool already claimed this role - skip this one
                    result.skipped.append(
                        (tool, f"role '{spec.role}' already claimed by another native tool")
                    )
                    continue
                result.claimed_roles.add(spec.role)

            if spec.hosted:
                result.hosted.append((tool, spec))
            else:
                result.native.append((tool, spec))

        # Second pass: process tools without native specs (generic function tools)
        for tool in tools:
            spec = self.resolve_native_spec(tool)
            if spec:
                # Already processed in first pass
                continue

            # Check if this tool's role is already claimed by a native tool
            tool_role = self.get_tool_role(tool)
            if tool_role and tool_role in result.claimed_roles:
                result.skipped.append((tool, f"role '{tool_role}' already claimed by native tool"))
                continue

            result.generic.append(tool)

        return result

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

        self.console.debug(
            f"Discovered {len(self._available_tools)} tools from environment: "
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
            if ctx.has_scenario:
                # Scenario was specified but prompt is still empty
                # (e.g., scenario returned empty string, or edge case not caught in scenarios.py)
                scenario = ctx._task.scenario if ctx._task else "unknown"
                raise ValueError(
                    f"ctx.prompt is not set.\n\n"
                    f"Scenario '{scenario}' was specified but returned an empty prompt.\n"
                    f"Check that the scenario's setup function returns a non-empty string."
                )
            else:
                # No scenario specified at all
                raise ValueError(
                    "ctx.prompt is not set.\n\n"
                    "No scenario was specified in your task file.\n"
                    "Either add a 'scenario' field to your task, or set ctx.prompt manually "
                    "before running the agent."
                )

        # Store context for tool calls
        self.ctx = ctx

        # Initialize tools from context
        if not self._initialized:
            await self._initialize_from_ctx(ctx)

        try:
            # Build initial context - optionally append setup tool output
            # Check ctx first (task-level override), then fall back to agent config
            append_setup = getattr(ctx, "append_setup_output", False) or getattr(
                self.config, "append_setup_output", False
            )
            initial_prompt = ctx.prompt
            if append_setup:
                setup_output = getattr(ctx, "setup_output", None)
                if setup_output:
                    initial_prompt = f"{initial_prompt}\n\n{setup_output}"

            # Build initial blocks (text prompt + optional screenshot)
            initial_blocks = text_to_blocks(initial_prompt)

            result = await self._run_context(initial_blocks, max_steps=max_steps)

            # Propagate error state to context for platform visibility
            if result.isError and hasattr(ctx, "error"):
                error_msg = result.info.get("error") if result.info else result.content
                ctx.error = Exception(str(error_msg)) if error_msg else Exception("Agent error")

            # Submit final answer to context (only if scenario is running)
            if result.content and ctx.has_scenario:
                await ctx.submit(result.content)

            return result

        except Exception as e:
            logger.exception("Error while running agent:")
            # Propagate error to context for platform visibility
            if hasattr(ctx, "error"):
                ctx.error = e
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
        # Use ctx.reward if already set (e.g., from scenario evaluate), otherwise 0.0
        # Note: For v4 tasks with evaluate_tool, reward is set in __aexit__ after this returns,
        # so callers should prefer ctx.reward over Trace.reward for the final result.
        reward = 0.0
        if self.ctx is not None:
            ctx_reward = getattr(self.ctx, "reward", None)
            if ctx_reward is not None:
                reward = ctx_reward

        trace_params = {
            "reward": reward,
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

    If isError is True, return 0.0 (error results should not contribute positive reward).
    If not found, return 0.0
    """
    # Error results should return 0.0 - don't extract reward from error responses
    if result.isError:
        logger.warning("Evaluate tool returned error, using reward=0.0")
        return 0.0

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

    logger.error("Couldn't parse reward from result: %s", str(result.structuredContent))
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
