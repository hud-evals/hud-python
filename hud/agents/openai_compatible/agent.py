"""OpenAI-compatible Chat Completions agent.

This class provides the minimal glue required to connect any endpoint that
implements the OpenAI-compatible *chat.completions* API with MCP tool calling
through the existing :class:`hud.agent.MCPAgent` scaffolding.

Key points:
- Stateless, no special server-side conversation state is assumed.
- Defaults to HUD inference gateway (inference.hud.ai) when HUD_API_KEY is set
- Accepts an :class:`openai.AsyncOpenAI` client, caller can supply their own
  base_url / api_key (e.g. llama.cpp, together.ai)
- All HUD features (step_count, OTel spans, tool filtering, screenshots)
  come from the ``MCPAgent`` base class, we only implement the three abstract
  methods
"""

from __future__ import annotations

import json
import logging
from typing import Any, cast

import mcp.types as types
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from hud.agents.base import AgentState, MCPAgent
from hud.agents.types import OpenAIChatConfig
from hud.settings import settings
from hud.types import AgentResponse, MCPToolCall
from hud.utils.types import with_signature

from .tools import (
    OpenAICompatibleAgentTools,
)

logger = logging.getLogger(__name__)


class OpenAIChatAgentState(AgentState[ChatCompletionMessageParam, OpenAICompatibleAgentTools]):
    continuation_token_ids: list[int] | None = None
    continuation_message_count: int | None = None


class OpenAIChatAgent(
    MCPAgent[ChatCompletionMessageParam, OpenAICompatibleAgentTools, OpenAIChatAgentState]
):
    """MCP-enabled agent that speaks the OpenAI *chat.completions* protocol."""

    @with_signature(OpenAIChatConfig)
    @classmethod
    def create(cls, **kwargs: Any) -> OpenAIChatAgent:  # pyright: ignore[reportIncompatibleMethodOverride]
        return cls(OpenAIChatConfig(**kwargs))

    def __init__(self, config: OpenAIChatConfig | None = None) -> None:
        config = config or OpenAIChatConfig()
        super().__init__(config)
        self.config: OpenAIChatConfig

        if (
            self.config.api_key
            and self.config.base_url
            and settings.hud_gateway_url in self.config.base_url
            and settings.api_key
            and self.config.api_key != settings.api_key
        ):
            raise ValueError(
                "OpenAIChatAgent api_key is not allowed with HUD Gateway. "
                "Use HUD_API_KEY for gateway auth and BYOK headers for provider keys."
            )

        self.oai: AsyncOpenAI
        if self.config.openai_client is not None:
            self.oai = self.config.openai_client
        elif self.config.api_key is not None or self.config.base_url is not None:
            self.oai = AsyncOpenAI(api_key=self.config.api_key, base_url=self.config.base_url)
        elif settings.api_key:
            # Default to HUD inference gateway
            self.oai = AsyncOpenAI(
                api_key=settings.api_key,
                base_url=settings.hud_gateway_url,
            )
        else:
            raise ValueError(
                "No API key found. Set HUD_API_KEY for HUD gateway, "
                "or provide api_key/base_url/openai_client explicitly."
            )

        self.completion_kwargs = dict(self.config.completion_kwargs)

        # If a specific checkpoint is requested, inject it into extra_body
        # so the HUD gateway routes to the exact checkpoint for inference.
        if self.config.checkpoint:
            extra_body: dict[str, Any] = dict(self.completion_kwargs.get("extra_body") or {})
            extra_body["checkpoint"] = self.config.checkpoint
            self.completion_kwargs["extra_body"] = extra_body

    async def initialize_state(self, prompt: list[types.PromptMessage]) -> OpenAIChatAgentState:
        """Format MCP prompt messages for OpenAI-compatible chat."""
        formatted_messages: list[ChatCompletionMessageParam] = []
        for message in prompt:
            content: list[dict[str, Any]] = []
            block = message.content
            if isinstance(block, types.TextContent):
                content.append({"type": "text", "text": block.text})
            elif isinstance(block, types.ImageContent):
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{block.mimeType};base64,{block.data}"},
                    }
                )

            formatted_messages.append(
                cast(
                    "ChatCompletionMessageParam",
                    {"role": message.role, "content": content},
                )
            )
        return OpenAIChatAgentState.model_construct(
            messages=formatted_messages,
            tools=OpenAICompatibleAgentTools(),
        )

    async def get_response(self, state: OpenAIChatAgentState) -> AgentResponse:
        """Send chat request to OpenAI and convert the response."""
        messages = state.messages

        reserved_kwargs = {"model", "messages", "stream", "tools"}
        request_kwargs = {
            key: value
            for key, value in self.completion_kwargs.items()
            if key not in reserved_kwargs
        }
        provider_body: dict[str, Any] = dict(request_kwargs.pop("extra_body", None) or {})
        return_token_ids = bool(provider_body.get("return_token_ids"))

        if state.tools.params:
            provider_body["tools"] = state.tools.params

        if return_token_ids and state.continuation_token_ids and state.continuation_message_count:
            provider_body["prompt_token_ids"] = state.continuation_token_ids
            provider_body["continuation_from"] = state.continuation_message_count

        if provider_body:
            request_kwargs["extra_body"] = provider_body

        try:
            response: ChatCompletion = await self.oai.chat.completions.create(
                model=self.config.model,
                messages=(
                    [{"role": "system", "content": self.system_prompt}, *messages]
                    if self.system_prompt is not None
                    else messages
                ),
                stream=False,
                **request_kwargs,
            )
        except Exception as e:
            error_content = f"Error getting response {e}"
            if "Invalid JSON" in str(e):
                error_content = "Invalid JSON, response was truncated"
            logger.warning(error_content)

            return AgentResponse(
                content=error_content,
                tool_calls=[],
                done=True,
                isError=True,
                raw=None,
            )

        choice = response.choices[0]
        message = choice.message
        function_calls = [
            tool_call for tool_call in message.tool_calls or [] if tool_call.type == "function"
        ]

        assistant_message = message.model_dump(exclude_none=True)
        reasoning_content = getattr(message, "reasoning_content", None)
        reasoning = reasoning_content if isinstance(reasoning_content, str) else None
        if not reasoning:
            raw_reasoning = getattr(message, "reasoning", None)
            reasoning = raw_reasoning if isinstance(raw_reasoning, str) else None
        for field in ("reasoning_content", "reasoning", "reasoning_details"):
            if value := getattr(message, field, None):
                assistant_message[field] = value
        if function_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in function_calls
            ]
        messages.append(cast("ChatCompletionMessageParam", assistant_message))

        if return_token_ids:
            prompt_token_ids = getattr(choice, "prompt_token_ids", None)
            token_ids = getattr(choice, "token_ids", None)
            if prompt_token_ids is not None and token_ids is not None:
                state.continuation_token_ids = list(prompt_token_ids) + list(token_ids)
                state.continuation_message_count = len(messages)

        tool_calls: list[MCPToolCall] = []
        for tool_call in function_calls:
            provider_name = tool_call.function.name
            raw_args = json.loads(tool_call.function.arguments or "{}")
            arguments = cast("dict[str, Any]", raw_args) if isinstance(raw_args, dict) else {}
            tool_calls.append(
                MCPToolCall(
                    id=tool_call.id,
                    name=state.tools.name_map.get(provider_name, provider_name),
                    arguments=arguments,
                )
            )

        return AgentResponse(
            content=message.content or "",
            reasoning=reasoning,
            info={"finish_reason": choice.finish_reason},
            tool_calls=tool_calls,
            done=not tool_calls,
            raw=response,
        )
