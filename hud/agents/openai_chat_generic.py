"""Generic OpenAI chat-completions agent.

This class provides the minimal glue required to connect any endpoint that
implements the OpenAI compatible *chat.completions* API with MCP tool calling
through the existing :class:`hud.agent.MCPAgent` scaffolding.

Key points:
- Stateless, no special server-side conversation state is assumed.
- Accepts an :class:`openai.AsyncOpenAI` client, caller can supply their own
  base_url / api_key (e.g. ART, llama.cpp, together.ai, …)
- All HUD features (step_count, OTel spans, tool filtering, screenshots, …)
  come from the ``MCPAgent`` base class, we only implement the three abstract
  methods
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, cast

from hud.agent import MCPAgent
from hud.types import AgentResponse, MCPToolCall, MCPToolResult

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletionToolParam

    from hud.clients import AgentMCPClient

logger = logging.getLogger(__name__)


class GenericOpenAIChatAgent(MCPAgent):
    """MCP-enabled agent that speaks the OpenAI *chat.completions* protocol."""

    def __init__(
        self,
        mcp_client: AgentMCPClient,
        *,
        openai_client: AsyncOpenAI,
        model_name: str = "gpt-4o-mini",
        parallel_tool_calls: bool = False,
        **agent_kwargs: Any,
    ) -> None:
        super().__init__(mcp_client=mcp_client, **agent_kwargs)
        self.oai = openai_client
        self.model_name = model_name
        self.parallel_tool_calls = parallel_tool_calls

    @staticmethod
    def _oai_to_mcp(tool_call: Any) -> MCPToolCall:  # type: ignore[valid-type]
        """Convert an OpenAI ``tool_call`` to :class:`MCPToolCall`."""
        return MCPToolCall(
            id=tool_call.id,
            name=tool_call.function.name,
            arguments=json.loads(tool_call.function.arguments or "{}"),
        )

    @staticmethod
    def _mcp_results_to_tool_messages(
        calls: list[MCPToolCall], results: list[MCPToolResult]
    ) -> list[dict[str, Any]]:
        """Render MCP tool results as OpenAI ``role=tool`` messages."""
        rendered: list[dict[str, Any]] = []
        for call, res in zip(calls, results, strict=False):
            if res.structuredContent:
                content = json.dumps(res.structuredContent)
            else:
                # Concatenate any TextContent blocks
                content = "".join(
                    c.text  # type: ignore[attr-defined]
                    for c in res.content
                    if hasattr(c, "text")
                )
            rendered.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": content or "",  # Ensure content is never None
                }
            )
        return rendered

    async def create_initial_messages(
        self, prompt: str, initial_screenshot: bool = False
    ) -> list[Any]:
        """Compose the starting conversation messages."""
        return [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": prompt},
        ]

    def get_tool_schemas(self) -> list[dict]:
        tool_schemas = super().get_tool_schemas()
        openai_tools = []
        for schema in tool_schemas:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": schema["name"],
                    "description": schema.get("description", ""),
                    "parameters": schema.get("parameters", {"type": "object", "properties": {}}),
                },
            }
            openai_tools.append(openai_tool)
        return openai_tools

    async def get_model_response(self, messages: list[Any]) -> AgentResponse:
        """Send chat request to OpenAI and convert the response."""
        # Convert MCP tool schemas to OpenAI format
        mcp_schemas = self.get_tool_schemas()

        response = await self.oai.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=cast("list[ChatCompletionToolParam]", mcp_schemas),
            parallel_tool_calls=self.parallel_tool_calls,
        )

        choice = response.choices[0]
        msg = choice.message

        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.function.name is not None:
                    tool_calls.append(self._oai_to_mcp(tc))
                    if not self.parallel_tool_calls:
                        break

        return AgentResponse(
            content=msg.content or "",
            tool_calls=tool_calls,
            done=choice.finish_reason == "stop",
            raw=response,  # Include raw response for access to Choice objects
        )

    async def format_tool_results(
        self,
        tool_calls: list[MCPToolCall],
        tool_results: list[MCPToolResult],
    ) -> list[Any]:
        """Format MCP results into OpenAI ``tool`` messages."""
        return self._mcp_results_to_tool_messages(tool_calls, tool_results)
