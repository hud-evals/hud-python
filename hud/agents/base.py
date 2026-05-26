"""Base MCP Agent implementation."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from hud.agents.misc import auto_respond
from hud.types import AgentResponse, Trace

if TYPE_CHECKING:
    import mcp.types as types

    from hud.agents.tools import AgentTools
    from hud.agents.tools.base import CallTool, ToolClient
    from hud.agents.types import AgentConfig

ProviderMessageT = TypeVar("ProviderMessageT")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentContext:
    """Prompt messages plus optional MCP tool access for one agent run."""

    messages: list[types.PromptMessage]
    tool_client: ToolClient | None = None


class MCPAgent(ABC, Generic[ProviderMessageT]):
    """
    Base class for agents that interact with HUD MCP-backed environments.

    Agent instances are intended to be run-scoped: create a fresh agent for each
    independent evaluation or task run. Provider implementations may keep
    conversation IDs, continuation cursors, and prepared tool state on the
    instance during a run.

    Agents interact with environments through per-run tools and tool handlers supplied
    by the caller.

    Subclasses implement provider-specific message formatting, response fetching,
    and tool result rendering.
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config

        self.model_name: str = self.config.model_name
        self.model: str = self.config.model

        self.system_prompt = self.config.system_prompt

        self.enable_citations: bool = False

        self.auto_respond: bool = config.auto_respond

    @classmethod
    def create(cls, **kwargs: object) -> MCPAgent[ProviderMessageT]:
        raise NotImplementedError(f"{cls.__name__}.create() must be implemented by subclasses")

    @cached_property
    @abstractmethod
    def tools(self) -> AgentTools[Any, Any]:
        """Provider-specific tool container used by the shared run loop."""
        raise NotImplementedError

    async def run(
        self,
        ctx: AgentContext,
        *,
        max_steps: int = 10,
    ) -> Trace:
        """
        Run the agent loop with prepared messages and optional tools.

        Args:
            ctx: Prompt messages and optional environment client
            max_steps: Maximum number of agent steps (-1 for infinite)

        Returns:
            Trace with reward, done, content fields and trace steps
        """
        tool_handler: CallTool | None = None
        if ctx.tool_client is not None:
            self.tools.prepare(
                model=self.model,
                tools=ctx.tool_client.tools,
                hosted_tools=self.config.hosted_tools,
                tool_metadata=ctx.tool_client.tool_metadata,
            )
            tool_handler = ctx.tool_client.tool_handler

        messages: list[ProviderMessageT] = []
        try:
            messages = await self.format_messages(ctx.messages)
            logger.debug("Messages: %s", messages)

            step_count = 0
            while max_steps == -1 or step_count < max_steps:
                step_count += 1
                if max_steps == -1:
                    logger.debug("Step %s (unlimited)", step_count)
                else:
                    logger.debug("Step %s/%s", step_count, max_steps)

                try:
                    # 1. Get model response
                    response = await self.get_response(messages)

                    logger.debug("Agent:\n%s", response)

                    if response.done or not response.tool_calls:
                        if follow_up := await auto_respond(
                            response.content,
                            enabled=self.auto_respond,
                        ):
                            logger.debug("Continuing execution")
                            messages.extend(await self.format_messages([follow_up]))
                            continue

                        logger.debug("Stopping execution")
                        return Trace(
                            done=True,
                            messages=messages,
                            content=response.content,
                            isError=response.isError,
                            citations=response.citations,
                        )

                    # 2. Execute tools
                    tool_messages = await self.tools.execute(
                        tool_handler,
                        response.tool_calls,
                    )

                    messages.extend(cast("list[ProviderMessageT]", tool_messages))

                except Exception as e:
                    logger.exception("Step failed")
                    return Trace(
                        done=True,
                        messages=messages,
                        content=str(e),
                        isError=True,
                        info={"error": str(e)},
                    )

        except KeyboardInterrupt:
            logger.warning("Agent execution interrupted by user")
            return Trace(
                done=True,
                messages=messages,
                content="Interrupted by user",
                isError=True,
                info={"error": "Interrupted by user"},
            )
        except asyncio.CancelledError:
            logger.warning("Agent execution cancelled")
            return Trace(
                done=True,
                messages=messages,
                content="Cancelled",
                isError=True,
                info={"error": "Cancelled"},
            )
        except Exception as e:
            logger.exception("Unexpected error")
            return Trace(
                done=True,
                messages=messages,
                content=str(e),
                isError=True,
                info={"error": str(e)},
            )

        return Trace(
            done=True,
            messages=messages,
        )

    @abstractmethod
    async def get_response(self, messages: list[ProviderMessageT]) -> AgentResponse:
        """
        Get response from the model including any tool calls.


        Args:
            messages: Current conversation messages

        Returns:
            AgentResponse with content, tool_calls, and done fields
        """
        raise NotImplementedError

    @abstractmethod
    async def format_messages(self, messages: list[types.PromptMessage]) -> list[ProviderMessageT]:
        """Format MCP prompt messages into provider messages."""
        raise NotImplementedError
