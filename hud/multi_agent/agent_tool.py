"""Agent-as-Tool pattern: Expose sub-agents as callable tools.

This module implements the following principle: Sub-agents are tools that the main
agent calls, not separate orchestration layers.

Key insight: "Don't communicate by sharing memory, share memory by communicating"
- Each sub-agent has its own isolated context window
- Only structured results return to main agent
- No intermediate noise pollutes main context
"""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, Any, TypeVar, Callable, ClassVar

from pydantic import BaseModel

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class SubAgentProtocol:
    """Protocol for SubAgent classes that can be decorated with @agent_as_tool."""

    _tool_name: ClassVar[str | None]
    _tool_description: ClassVar[str | None]
    _return_schema: ClassVar[type[BaseModel] | None]
    _tool_schema: ClassVar[dict[str, Any] | None]


def agent_as_tool(
    name: str | None = None,
    description: str | None = None,
    returns: type[T] | None = None,
) -> Callable[[type[Any]], type[Any]]:
    """Decorator to expose a sub-agent as a tool.

    When applied to a SubAgent class, this decorator:
    1. Creates a tool definition from the agent's signature
    2. Runs the agent in an isolated context window
    3. Returns a SubAgentResult (token-optimized)

    Example:
        @agent_as_tool(name="research", returns=ResearchResult)
        class ResearcherAgent(SubAgent):
            '''Deep research on a topic.'''
            pass

        # Later, main agent can call:
        # result = await call_tool(name="research", arguments={"query": "..."})

    Args:
        name: Tool name (defaults to class name in snake_case)
        description: Tool description (defaults to class docstring)
        returns: Return schema for documentation/hints only. Sub-agents always
            return SubAgentResult format with fields: output, success, error,
            artifacts, summary, log_file, duration_ms. The 'returns' schema
            is NOT used for validation to avoid data loss.

    Returns:
        Decorated class with tool metadata
    """

    def decorator(cls: type[Any]) -> type[Any]:
        # Extract name from class if not provided
        tool_name = name
        if tool_name is None:
            # Convert CamelCase to snake_case and remove "Agent" suffix
            class_name = cls.__name__
            if class_name.endswith("Agent"):
                class_name = class_name[:-5]
            tool_name = _camel_to_snake(class_name)

        # Extract description from docstring if not provided
        tool_description = description
        if tool_description is None:
            tool_description = inspect.getdoc(cls) or f"Run {cls.__name__}"

        # Store metadata on the class
        cls._tool_name = tool_name
        cls._tool_description = tool_description
        cls._return_schema = returns

        # Create the as_tool class method and attach it
        async def _as_tool_call(
            agent_cls: type[Any],
            ctx: Any,
            **kwargs: Any,
        ) -> T | dict[str, Any]:
            """Execute this agent as a tool call.

            Args:
                agent_cls: The agent class
                ctx: Parent EvalContext (for MCP access)
                **kwargs: Arguments passed to the agent (including run_id)

            Returns:
                Structured result (return schema instance or dict)
            """
            # Extract run_id from kwargs (passed from parent runner)
            run_id = kwargs.pop("run_id", None)

            # Create agent instance with isolated context
            agent = agent_cls(
                isolation=True,
                parent_ctx=ctx,
                run_id=run_id,  # Use parent's run_id for unified logging
            )

            # Run with provided arguments as prompt
            prompt = kwargs.get("prompt") or kwargs.get("query") or str(kwargs)
            result = await agent.run_isolated(prompt, **kwargs)

            # Return result directly - sub-agents return SubAgentResult format
            # which is token-optimized.
            #
            # NOTE: We no longer try to validate against the configured return_schema
            # (e.g., CodeResult, ResearchResult) because:
            # 1. SubAgentResult has different fields than these schemas
            # 2. Since all schema fields have defaults, Pydantic silently succeeds
            #    but loses important fields like 'output', 'artifacts', 'summary', 'log_file'
            # 3. The return_schema is for documentation/type hints only
            #
            # If you need structured output from sub-agents matching a specific schema,
            # the sub-agent should be prompted to return that structure, and you can
            # validate the 'output' field content rather than the SubAgentResult wrapper.
            return result

        # Attach the method to the class
        setattr(cls, "as_tool_call", classmethod(lambda c, ctx, **kw: _as_tool_call(c, ctx, **kw)))

        # Create tool schema for MCP registration
        input_schema = _create_input_schema(cls)

        cls._tool_schema = {
            "name": tool_name,
            "description": tool_description,
            "inputSchema": input_schema,
        }

        return cls

    return decorator


def _camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case."""
    import re

    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _create_input_schema(cls: type) -> dict[str, Any]:
    """Create JSON schema for tool input from agent class.

    Looks for:
    1. Class-level input_schema attribute
    2. run_isolated method signature
    3. Default schema with query parameter
    """
    # Check for explicit schema
    if hasattr(cls, "input_schema"):
        return cls.input_schema

    # Try to infer from run_isolated signature
    if hasattr(cls, "run_isolated"):
        sig = inspect.signature(cls.run_isolated)
        properties = {}
        required = []

        for name, param in sig.parameters.items():
            if name in ("self", "prompt"):
                continue

            # Determine type
            param_type = "string"
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"

            properties[name] = {"type": param_type}

            # Check if required
            if param.default == inspect.Parameter.empty:
                required.append(name)

        # Always include query/prompt
        if "query" not in properties and "prompt" not in properties:
            properties["query"] = {
                "type": "string",
                "description": "The query or prompt for this agent",
            }
            required.append("query")

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    # Default schema
    return {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The query or prompt for this agent",
            }
        },
        "required": ["query"],
    }


class AgentToolRegistry:
    """Registry of agents exposed as tools.

    Allows main agent to discover and call sub-agents as tools.
    """

    def __init__(self) -> None:
        self._agents: dict[str, type[Any]] = {}

    def register(self, agent_cls: type[Any]) -> None:
        """Register an agent class as a tool.

        Args:
            agent_cls: SubAgent class decorated with @agent_as_tool
        """
        if not hasattr(agent_cls, "_tool_name"):
            raise ValueError(f"{agent_cls.__name__} must be decorated with @agent_as_tool")

        tool_name = getattr(agent_cls, "_tool_name", None)
        if tool_name is None:
            raise ValueError(f"{agent_cls.__name__} has no _tool_name")

        self._agents[tool_name] = agent_cls
        logger.info(f"Registered agent tool: {tool_name}")

    def get(self, name: str) -> type[Any] | None:
        """Get an agent class by tool name."""
        return self._agents.get(name)

    def list_tools(self) -> list[dict[str, Any]]:
        """Get tool schemas for all registered agents."""
        schemas = []
        for cls in self._agents.values():
            schema = getattr(cls, "_tool_schema", None)
            if schema is not None:
                schemas.append(schema)
        return schemas

    def get_tool_names(self) -> list[str]:
        """Get names of all registered agent tools."""
        return list(self._agents.keys())

    async def call(
        self,
        name: str,
        ctx: Any,
        **kwargs: Any,
    ) -> Any:
        """Call an agent tool by name.

        Args:
            name: Tool name
            ctx: Parent EvalContext
            **kwargs: Arguments to pass to agent

        Returns:
            Structured result from agent
        """
        agent_cls = self._agents.get(name)
        if agent_cls is None:
            raise ValueError(f"Unknown agent tool: {name}")

        # Call the as_tool_call method
        as_tool_call = getattr(agent_cls, "as_tool_call", None)
        if as_tool_call is None:
            raise ValueError(f"Agent {name} has no as_tool_call method")

        return await as_tool_call(ctx, **kwargs)


# Global registry instance
agent_tools = AgentToolRegistry()


def register_agent_tool(cls: type[Any]) -> type[Any]:
    """Convenience decorator to register agent with global registry.

    Example:
        @register_agent_tool
        @agent_as_tool(name="research", returns=ResearchResult)
        class ResearcherAgent(SubAgent):
            pass
    """
    agent_tools.register(cls)
    return cls


__all__ = [
    "agent_as_tool",
    "AgentToolRegistry",
    "agent_tools",
    "register_agent_tool",
]

