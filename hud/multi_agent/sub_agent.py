"""SubAgent with isolated context window.

This module implements sub-agents that run in their own context window,
following the following principle: context isolation:
- Each sub-agent has its own conversation history
- Only minimal structured results return to parent (token optimized)
- Detailed execution trace written to per-agent YAML log file
- No intermediate noise pollutes parent context
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from hud.multi_agent.context import AppendOnlyContext
from hud.multi_agent.logger import SubAgentLogger, _sanitize_tool_results
from hud.multi_agent.schemas import GenericResult, SubAgentResult

if TYPE_CHECKING:
    from hud.eval.context import EvalContext

logger = logging.getLogger(__name__)


class SubAgentConfig(BaseModel):
    """Configuration for a sub-agent."""

    name: str = "sub_agent"
    model: str = "anthropic/claude-sonnet-4-5-20250929"
    max_steps: int = 10
    timeout: int = 300  # seconds

    # Context isolation
    isolation: bool = True  # Each sub-agent has own context
    max_context_tokens: int = 32_000  # Smaller than main agent

    # System prompt
    system_prompt: str = ""

    # Tools available to this agent
    tools: list[str] = []

    # Return schema (string name of schema class)
    return_schema: str | None = None

    # Claude-specific settings
    use_computer_beta: bool = True  # Use Anthropic computer_use beta format

    # Logging settings
    log_dir: str = ".logs"  # Directory for execution logs


class SubAgent:
    """Base class for sub-agents with isolated context.

    Sub-agents:
    1. Have their own AppendOnlyContext (isolated from parent)
    2. Can access parent's tools via parent_ctx
    3. Return only structured results to parent

    Example:
        @agent_as_tool(name="research", returns=ResearchResult)
        class ResearcherAgent(SubAgent):
            async def run_isolated(self, prompt: str, **kwargs) -> dict:
                # Research logic here
                return {
                    "summary": "...",
                    "sources": [...],
                    "confidence": 0.9
                }
    """

    # Class-level metadata (set by @agent_as_tool decorator)
    _tool_name: str | None = None
    _tool_description: str | None = None
    _return_schema: type[BaseModel] | None = None
    _tool_schema: dict[str, Any] | None = None

    def __init__(
        self,
        config: SubAgentConfig | None = None,
        isolation: bool = True,
        parent_ctx: EvalContext | None = None,
        run_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the sub-agent.

        Args:
            config: Agent configuration
            isolation: Whether to use isolated context (default: True)
            parent_ctx: Parent EvalContext for tool access
            run_id: Run ID for logging (auto-generated if not provided)
            **kwargs: Additional config overrides
        """
        from uuid import uuid4

        # Build config
        if config is None:
            config = SubAgentConfig(**kwargs)
        self.config = config

        self.isolation = isolation
        self.parent_ctx = parent_ctx
        self.run_id = run_id or uuid4().hex[:12]

        # Create isolated context if needed
        if self.isolation:
            self.context = AppendOnlyContext(max_tokens=config.max_context_tokens)
        else:
            self.context = None

        # Track execution
        self._start_time: datetime | None = None
        self._step_count = 0
        self._sub_logger: SubAgentLogger | None = None
        self._artifacts: list[str] = []  # Track files created/modified

    @property
    def name(self) -> str:
        """Get agent name."""
        return self._tool_name or self.config.name

    async def run_isolated(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Run the agent in isolated context.

        Override this method to implement agent logic.

        Args:
            prompt: The task prompt
            **kwargs: Additional arguments

        Returns:
            Dict conforming to return schema
        """
        # Default implementation - subclasses should override
        return await self._default_run(prompt, **kwargs)

    async def _default_run(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Default run implementation using LLM.

        This provides a basic implementation that:
        1. Sets up isolated context
        2. Calls LLM with available tools
        3. Returns structured result
        """
        import time

        self._start_time = datetime.now()
        start = time.time()

        try:
            # Initialize context
            if self.context:
                if self.config.system_prompt:
                    self.context.append_system(self.config.system_prompt)
                self.context.append_user(prompt)
                self.context.freeze_prefix()

            # If we have parent context, we can use its tools
            if self.parent_ctx is not None:
                result = await self._run_with_tools(prompt, **kwargs)
            else:
                # No tools - just return the prompt as result
                result = GenericResult(
                    output=f"SubAgent received: {prompt}",
                    data=kwargs,
                ).model_dump()

            # Calculate duration
            duration_ms = (time.time() - start) * 1000

            # Add metadata
            if isinstance(result, dict):
                result["duration_ms"] = duration_ms
                result["success"] = True

            return result

        except Exception as e:
            logger.exception(f"SubAgent {self.name} failed:")
            duration_ms = (time.time() - start) * 1000

            return {
                "success": False,
                "error": str(e),
                "duration_ms": duration_ms,
            }

    async def _run_with_tools(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Run with access to parent's tools using a real LLM agent.

        Creates and runs an actual LLM agent (ClaudeAgent or OpenAIAgent) based
        on the config model, giving it access to the parent context's tools.

        Returns minimal SubAgentResult format (token optimized):
        - output: Natural language summary
        - success: bool
        - error: Error message if failed (KEPT)
        - artifacts: List of file paths created/modified
        - summary: Brief action summary
        - log_file: Path to detailed YAML execution log
        """
        import time

        # Get available tools from parent
        if self.parent_ctx is None:
            return {"output": prompt, "error": "No parent context", "success": False}

        start_time = time.time()

        try:
            logger.info(f"SubAgent [{self.name}] starting")

            # Initialize sub-agent logger
            # Determine invocation number by counting existing log files for this agent
            run_dir = Path(self.config.log_dir) / self.run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            # Count existing log files for this agent to get invocation number
            existing_logs = list(run_dir.glob(f"{self.name}_*.yaml"))
            invocation = len(existing_logs) + 1

            self._sub_logger = SubAgentLogger(
                run_dir=run_dir,
                agent_name=self.name,
                prompt=prompt,
                invocation=invocation,
            )

            # Log available tools for debugging
            tools = await self.parent_ctx.list_tools()
            tool_names = [t.name for t in tools]

            # Filter to allowed tools if specified in config
            if self.config.tools:
                allowed_tools = [t for t in tool_names if t in self.config.tools]
                logger.debug(f"SubAgent [{self.name}] filtered tools: {allowed_tools}")
            else:
                allowed_tools = tool_names

            # Create the appropriate LLM agent based on model config
            model = self.config.model
            system_prompt = self.config.system_prompt or f"You are a specialist agent: {self.name}"

            # Choose agent based on model prefix
            if model.startswith("anthropic/") or "claude" in model.lower():
                from hud.agents.claude import ClaudeAgent

                agent = ClaudeAgent.create(
                    ctx=self.parent_ctx,
                    model=model.replace("anthropic/", ""),
                    system_prompt=system_prompt,
                    use_computer_beta=self.config.use_computer_beta,
                )
            elif model.startswith("openai/") or "gpt" in model.lower():
                from hud.agents.openai import OpenAIAgent

                agent = OpenAIAgent.create(
                    ctx=self.parent_ctx,
                    model=model.replace("openai/", ""),
                    system_prompt=system_prompt,
                )
            else:
                # Default to Claude for anthropic models, OpenAI otherwise
                if (
                    "sonnet" in model.lower()
                    or "opus" in model.lower()
                    or "haiku" in model.lower()
                ):
                    from hud.agents.claude import ClaudeAgent

                    agent = ClaudeAgent.create(
                        ctx=self.parent_ctx,
                        model=model,
                        system_prompt=system_prompt,
                        use_computer_beta=self.config.use_computer_beta,
                    )
                else:
                    from hud.agents.openai import OpenAIAgent

                    agent = OpenAIAgent.create(
                        ctx=self.parent_ctx,
                        model=model,
                        system_prompt=system_prompt,
                    )

            logger.debug(
                f"SubAgent [{self.name}] using model: {model}, max_steps: {self.config.max_steps}"
            )

            # Set the prompt on the parent context temporarily
            original_prompt = getattr(self.parent_ctx, "prompt", None)
            self.parent_ctx.prompt = prompt

            try:
                # Store context for the agent
                agent.ctx = self.parent_ctx

                # Manually initialize tools from context
                if not agent._initialized:
                    await agent._initialize_from_ctx(self.parent_ctx)

                # Now filter the agent's tools to only allowed tools (exclude other sub-agents)
                if self.config.tools:
                    # Filter to only the tools specified in config
                    original_tools = agent._available_tools or []
                    filtered_tools = [t for t in original_tools if t.name in self.config.tools]
                    agent._available_tools = filtered_tools
                    agent._tool_map = {t.name: t for t in filtered_tools}
                    # Also update the tool conversion for provider-specific formats
                    if hasattr(agent, "_on_tools_ready"):
                        agent._on_tools_ready()
                else:
                    # Even if no specific tools configured, exclude sub-agent tools
                    subagent_names = {"coder", "researcher", "reviewer"}
                    original_tools = agent._available_tools or []
                    filtered_tools = [t for t in original_tools if t.name not in subagent_names]
                    agent._available_tools = filtered_tools
                    agent._tool_map = {t.name: t for t in filtered_tools}
                    if hasattr(agent, "_on_tools_ready"):
                        agent._on_tools_ready()

                # Run the agent (skip initialization since we already did it)
                from hud.agents.base import text_to_blocks

                trace = await agent._run_context(
                    text_to_blocks(prompt), max_steps=self.config.max_steps
                )

                # Extract tool calls from messages and log to sub-agent YAML
                tool_calls = []
                tool_results = []
                tool_call_map: dict[str, dict[str, Any]] = {}  # id -> tool call info

                for msg in trace.messages:
                    # Handle Claude-style messages (list of content blocks)
                    if isinstance(msg, dict):
                        content = msg.get("content", [])
                        role = msg.get("role", "")

                        # Claude format: content is a list of blocks
                        if isinstance(content, list):
                            for block in content:
                                # Handle dict blocks
                                if isinstance(block, dict):
                                    block_type = block.get("type")
                                    if block_type == "tool_use":
                                        tc_info = {
                                            "name": block.get("name"),
                                            "arguments": block.get("input", {}),
                                            "id": block.get("id"),
                                        }
                                        tool_calls.append(tc_info)
                                        if tc_info["id"]:
                                            tool_call_map[tc_info["id"]] = tc_info
                                    elif block_type == "tool_result":
                                        tr_info = {
                                            "tool_use_id": block.get("tool_use_id"),
                                            "content": block.get("content"),
                                        }
                                        tool_results.append(tr_info)
                                        # Log to sub-agent YAML
                                        self._log_tool_execution(tr_info, tool_call_map)
                                # Handle Pydantic model blocks (Claude SDK)
                                elif hasattr(block, "type"):
                                    block_type = getattr(block, "type", None)
                                    if block_type == "tool_use":
                                        tc_info = {
                                            "name": getattr(block, "name", None),
                                            "arguments": getattr(block, "input", {}),
                                            "id": getattr(block, "id", None),
                                        }
                                        tool_calls.append(tc_info)
                                        if tc_info["id"]:
                                            tool_call_map[tc_info["id"]] = tc_info
                                    elif block_type == "tool_result":
                                        tr_info = {
                                            "tool_use_id": getattr(block, "tool_use_id", None),
                                            "content": getattr(block, "content", None),
                                        }
                                        tool_results.append(tr_info)
                                        self._log_tool_execution(tr_info, tool_call_map)

                        # OpenAI format: tool_calls is a list
                        if "tool_calls" in msg:
                            for tc in msg["tool_calls"]:
                                if isinstance(tc, dict):
                                    func = tc.get("function", {})
                                    tc_info = {
                                        "name": func.get("name"),
                                        "arguments": func.get("arguments", {}),
                                        "id": tc.get("id"),
                                    }
                                    tool_calls.append(tc_info)
                                    if tc_info["id"]:
                                        tool_call_map[tc_info["id"]] = tc_info

                        # OpenAI format: tool role message
                        if role == "tool":
                            tr_info = {
                                "tool_call_id": msg.get("tool_call_id"),
                                "content": content if isinstance(content, str) else str(content),
                            }
                            tool_results.append(tr_info)
                            self._log_tool_execution(tr_info, tool_call_map)

                    # Handle Pydantic model messages (from Claude SDK)
                    elif hasattr(msg, "content"):
                        content = msg.content
                        if isinstance(content, list):
                            for block in content:
                                if hasattr(block, "type"):
                                    if block.type == "tool_use":
                                        tc_info = {
                                            "name": getattr(block, "name", None),
                                            "arguments": getattr(block, "input", {}),
                                            "id": getattr(block, "id", None),
                                        }
                                        tool_calls.append(tc_info)
                                        if tc_info["id"]:
                                            tool_call_map[tc_info["id"]] = tc_info
                                    elif block.type == "tool_result":
                                        tr_info = {
                                            "tool_use_id": getattr(block, "tool_use_id", None),
                                            "content": getattr(block, "content", None),
                                        }
                                        tool_results.append(tr_info)
                                        self._log_tool_execution(tr_info, tool_call_map)

                # Extract artifacts from tool calls (file paths created/modified)
                artifacts = self._extract_artifacts(tool_calls)

                # Finalize sub-agent log and get log file path
                duration_ms = (time.time() - start_time) * 1000
                error_msg = trace.info.get("error") if trace.isError and trace.info else None

                log_file = ""
                if self._sub_logger:
                    log_file = self._sub_logger.finalize(
                        output=trace.content or "",
                        success=not trace.isError,
                        error=error_msg,
                        artifacts=artifacts,
                    )

                # Build MINIMAL result (token optimized)
                # NO tool_calls or tool_results arrays - they're in the log file
                tool_names = [tc.get("name", "unknown") for tc in tool_calls]
                result = SubAgentResult(
                    output=trace.content or "",
                    success=not trace.isError,
                    error=error_msg,
                    artifacts=artifacts,
                    summary=f"Executed {len(tool_calls)} tools: {', '.join(tool_names[:5])}"
                    + ("..." if len(tool_names) > 5 else ""),
                    log_file=log_file,
                    duration_ms=duration_ms,
                ).model_dump()

                # Log to isolated context if we have one
                if self.context:
                    self.context.append_assistant(trace.content or "")

                logger.debug(
                    f"SubAgent [{self.name}] completed with {len(tool_calls)} tool calls"
                )
                return result

            finally:
                # Restore original prompt
                if original_prompt is not None:
                    self.parent_ctx.prompt = original_prompt

        except Exception as e:
            logger.exception(f"SubAgent {self.name} failed:")
            duration_ms = (time.time() - start_time) * 1000

            # Finalize log with error
            log_file = ""
            if self._sub_logger:
                log_file = self._sub_logger.finalize(
                    output="",
                    success=False,
                    error=str(e),
                    artifacts=[],
                )

            return SubAgentResult(
                output="",
                success=False,
                error=str(e),  # KEEP errors visible
                artifacts=[],
                summary="Failed with error",
                log_file=log_file,
                duration_ms=duration_ms,
            ).model_dump()

    def _log_tool_execution(
        self, tool_result: dict[str, Any], tool_call_map: dict[str, dict[str, Any]]
    ) -> None:
        """Log a tool execution to the sub-agent YAML log.

        Args:
            tool_result: Tool result dict with tool_use_id/tool_call_id and content
            tool_call_map: Map of tool call IDs to tool call info
        """
        if not self._sub_logger:
            return

        # Find the corresponding tool call
        tool_id = tool_result.get("tool_use_id") or tool_result.get("tool_call_id")
        tool_call = tool_call_map.get(tool_id, {}) if tool_id else {}

        tool_name = tool_call.get("name", "unknown")
        arguments = tool_call.get("arguments", {})

        # Extract result content
        result_content = tool_result.get("content", "")
        if isinstance(result_content, list):
            # Extract text from content blocks
            text_parts = []
            for item in result_content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)
            result_content = "\n".join(text_parts)
        elif not isinstance(result_content, str):
            result_content = str(result_content)

        self._sub_logger.log_execution_step(
            tool=tool_name,
            arguments=arguments,
            result=result_content,
        )

    def _extract_artifacts(self, tool_calls: list[dict[str, Any]]) -> list[str]:
        """Extract file paths from tool calls that create/modify files.

        Args:
            tool_calls: List of tool call dicts

        Returns:
            List of file paths
        """
        artifacts = []
        file_creating_tools = {
            "str_replace_based_edit_tool",
            "write_file",
            "create_file",
            "bash",
        }

        for tc in tool_calls:
            name = tc.get("name", "")
            args = tc.get("arguments", {})

            if name in file_creating_tools:
                # Extract path from arguments
                path = args.get("path") or args.get("file_path") or args.get("filename")
                if path and path not in artifacts:
                    artifacts.append(path)

                # For bash commands, try to extract output file paths
                if name == "bash":
                    command = args.get("command", "")
                    # Simple heuristic: look for ">" redirect or common patterns
                    if ">" in command:
                        parts = command.split(">")
                        if len(parts) > 1:
                            output_file = parts[-1].strip().split()[0]
                            if output_file and output_file not in artifacts:
                                artifacts.append(output_file)

        return artifacts

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool via parent context.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool result
        """
        from hud.types import MCPToolCall

        if self.parent_ctx is None:
            raise RuntimeError("No parent context - cannot call tools")

        # Log to isolated context
        if self.context:
            self.context.append_tool_call(name, arguments, agent_id=self.name)

        # Execute via parent - use MCPToolCall for compatibility with Environment.call_tool
        tool_call = MCPToolCall(name=name, arguments=arguments)
        result = await self.parent_ctx.call_tool(tool_call)

        # Log result to isolated context
        if self.context:
            content = str(result.content) if hasattr(result, "content") else str(result)
            self.context.append_tool_result(content, tool_name=name, agent_id=self.name)

        return result

    def get_context_snapshot(self) -> dict[str, Any]:
        """Get snapshot of isolated context for logging."""
        if self.context:
            return self.context.snapshot()
        return {}

    @classmethod
    def get_tool_schema(cls) -> dict[str, Any] | None:
        """Get the tool schema for this agent."""
        return cls._tool_schema

    @classmethod
    def get_return_schema(cls) -> type[BaseModel] | None:
        """Get the return schema class."""
        return cls._return_schema


class SimpleSubAgent(SubAgent):
    """A simple sub-agent that wraps a function.

    Use this for quick prototyping without creating a full class.

    Example:
        async def research_fn(prompt: str, **kwargs) -> dict:
            return {"summary": f"Researched: {prompt}"}

        agent = SimpleSubAgent(
            name="researcher",
            fn=research_fn,
            return_schema=ResearchResult,
        )
    """

    def __init__(
        self,
        name: str,
        fn: Any,  # Callable[[str, ...], Awaitable[dict]]
        return_schema: type[BaseModel] | None = None,
        description: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._fn = fn
        self._tool_name = name
        self._tool_description = description or fn.__doc__ or f"Run {name}"
        self._return_schema = return_schema

    async def run_isolated(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Run the wrapped function."""
        import asyncio

        if asyncio.iscoroutinefunction(self._fn):
            return await self._fn(prompt, **kwargs)
        else:
            return self._fn(prompt, **kwargs)


__all__ = ["SubAgent", "SubAgentConfig", "SimpleSubAgent"]

