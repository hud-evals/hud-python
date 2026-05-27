"""Agent ABC + legacy MCPAgent.

``Agent`` (new) is the minimal contract: declare which CapabilityClient
classes the agent natively drives, ``initialize(manifest)`` to negotiate /
open clients, ``run(...)`` to execute the scenario, ``close()`` to clean up.
Subclasses define their own ``run`` signature (a ToolAgent takes a prompt
and max_steps; a robotics agent takes a goal pose and a control rate; ...).

``MCPAgent`` (legacy) is the previous MCP-server-coupled base; it stays
here while we port the provider implementations (Claude / Gemini / OpenAI)
to ``Agent``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, ConfigDict

from hud.agents.misc import auto_respond
from hud.types import AgentResponse, Trace

if TYPE_CHECKING:
    import mcp.types as types

    from hud.agents.tools import AgentTools
    from hud.agents.tools.base import CallTool, ToolClient
    from hud.agents.types import AgentConfig
    from hud.capabilities import CapabilityClient
    from hud.client import Manifest

MessageT = TypeVar("MessageT")
ToolsT = TypeVar("ToolsT", bound="AgentTools[Any, Any, Any]")
logger = logging.getLogger(__name__)


# ─────────────────────────── Agent ABC (new) ───────────────────────────


class Agent(ABC):
    """Minimal agent contract.

    Lifecycle mirrors the wire:

    * ``initialize(manifest)`` — capability negotiation. Reconcile the env's
      published bindings against ``type(self).clients`` and open the matching
      clients; cache them on ``self`` for ``run`` to use.
    * ``run(...)`` — scenario execution. Subclasses define the signature
      (prompt + max_steps for tool agents; goal + rate for robotics; etc.).
    * ``close()`` — release the opened clients.
    """

    #: Static — CapabilityClient classes this agent type can drive.
    clients: ClassVar[tuple[type[CapabilityClient], ...]] = ()

    #: Populated by ``initialize``; ``{binding_name: opened_client}``.
    connections: dict[str, CapabilityClient]

    async def initialize(self, manifest: Manifest) -> None:
        """Open clients for every manifest binding whose protocol we support, in parallel.

        Subclasses can override (calling ``super().initialize(manifest)``) to
        add provider-specific state (LLM client, model config, etc.).
        """
        by_protocol = {cls.protocol: cls for cls in type(self).clients}
        pairs = [
            (b, by_protocol[b.protocol])
            for b in manifest.bindings
            if b.protocol in by_protocol
        ]
        opened = await asyncio.gather(*(cls.connect(b) for b, cls in pairs))
        self.connections = {
            b.name: c for (b, _), c in zip(pairs, opened, strict=False)
        }

    # Subclasses define their own ``run`` signature. There's no universal
    # contract — a ToolAgent runs ``run(*, prompt, max_steps)``, a robotics
    # agent runs ``run(*, goal, control_hz)``, a training agent runs
    # ``run(*, dataset, epochs)``. All return a ``Trace``.

    async def close(self) -> None:
        """Close every opened client. Idempotent; safe to call without initialize."""
        for client in getattr(self, "connections", {}).values():
            with contextlib.suppress(Exception):
                await client.close()
        self.connections = {}


class AgentState(BaseModel, Generic[MessageT, ToolsT]):
    """Mutable provider-formatted state for one agent run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: list[MessageT]
    tools: ToolsT


StateT = TypeVar("StateT", bound="AgentState[Any, Any]")


@dataclass
class AgentContext(Generic[StateT]):
    """Prompt input, tools, and run-local options for one agent run."""

    prompt: list[types.PromptMessage]
    tool_client: ToolClient | None = None
    # Per-run override; falls back to AgentConfig.system_prompt.
    system_prompt: str | None = None
    citations_enabled: bool = False
    state: StateT | None = None


class MCPAgent(ABC, Generic[MessageT, ToolsT, StateT]):
    """
    Base class for agents that interact with HUD MCP-backed environments.

    Agent instances hold provider configuration and clients. Per-run messages
    and provider state live on ``AgentContext`` under the ``state`` field.

    Agents interact with environments through per-run tools and tool handlers supplied
    by the caller.

    Subclasses implement provider-specific message formatting, response fetching,
    and tool result rendering.
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config

        self.model_name: str = self.config.model_name
        self.model: str = self.config.model

        self.auto_respond: bool = config.auto_respond

    @classmethod
    def create(cls, **kwargs: object) -> MCPAgent[MessageT, ToolsT, StateT]:
        raise NotImplementedError(f"{cls.__name__}.create() must be implemented by subclasses")

    async def run(
        self,
        ctx: AgentContext[StateT],
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
        if max_steps < -1:
            raise ValueError("max_steps must be -1 or greater")

        tool_handler: CallTool | None = None
        tools: list[types.Tool] = []
        if ctx.tool_client is not None:
            tools = ctx.tool_client.tools
            tool_handler = ctx.tool_client.tool_handler

        messages: list[MessageT] = []
        system_prompt = (
            ctx.system_prompt if ctx.system_prompt is not None else self.config.system_prompt
        )
        citations_enabled = ctx.citations_enabled
        try:
            state = await self.initialize_state(ctx.prompt)
            ctx.state = state
            state.tools.prepare(
                model=self.model,
                tools=tools,
                hosted_tools=self.config.hosted_tools,
            )
            messages = state.messages
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
                    response = await self.get_response(
                        state,
                        system_prompt=system_prompt,
                        citations_enabled=citations_enabled,
                    )

                    logger.debug("Agent:\n%s", response)

                    if response.done or not response.tool_calls:
                        if follow_up := await auto_respond(
                            response.content,
                            enabled=self.auto_respond,
                        ):
                            logger.debug("Continuing execution")
                            follow_up_state = await self.initialize_state([follow_up])
                            state.messages.extend(follow_up_state.messages)
                            continue

                        logger.debug("Stopping execution")
                        return Trace(
                            done=True,
                            messages=state.messages,
                            content=response.content,
                            isError=response.isError,
                            citations=response.citations,
                        )

                    # 2. Execute tools
                    tool_messages = await state.tools.execute(
                        tool_handler,
                        response.tool_calls,
                    )

                    state.messages.extend(tool_messages)

                except Exception as e:
                    logger.exception("Step failed")
                    return Trace(
                        done=True,
                        messages=state.messages,
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
            content="Max steps exceeded",
            isError=True,
            info={"error": "max_steps_exceeded", "max_steps": max_steps},
        )

    @abstractmethod
    async def get_response(
        self,
        state: StateT,
        *,
        system_prompt: str | None = None,
        citations_enabled: bool = False,
    ) -> AgentResponse:
        """
        Get response from the model including any tool calls.


        Args:
            state: Current provider conversation state
            system_prompt: Resolved run system prompt, if any
            citations_enabled: Whether provider citation metadata should be requested

        Returns:
            AgentResponse with content, tool_calls, and done fields
        """
        raise NotImplementedError

    @abstractmethod
    async def initialize_state(
        self,
        prompt: list[types.PromptMessage],
    ) -> StateT:
        """Build provider run state from MCP prompt messages."""
        raise NotImplementedError
