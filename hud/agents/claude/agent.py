"""Claude MCP Agent implementation."""

from __future__ import annotations

import copy
import json
import logging
from typing import TYPE_CHECKING, Literal, cast

import mcp.types as mcp_types
from anthropic import AsyncAnthropic, AsyncAnthropicBedrock, Omit
from anthropic.types import CacheControlEphemeralParam
from anthropic.types.beta import (
    BetaBase64ImageSourceParam,
    BetaBase64PDFSourceParam,
    BetaImageBlockParam,
    BetaMessage,
    BetaMessageParam,
    BetaRequestDocumentBlockParam,
    BetaTextBlock,
    BetaTextBlockParam,
    BetaToolChoiceAutoParam,
    BetaToolUnionParam,
)

from hud.agents import gateway
from hud.agents.base import AgentState, MCPAgent
from hud.agents.types import ClaudeConfig
from hud.settings import settings
from hud.tools.types import Citation
from hud.types import AgentResponse, MCPToolCall
from hud.utils.types import with_signature

from .tools import ClaudeAgentTools

if TYPE_CHECKING:
    import mcp.types as types
    from anthropic.types.beta import BetaTextCitation

logger = logging.getLogger(__name__)
ClaudeImageMediaType = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]


class ClaudeAgentState(AgentState[BetaMessageParam, ClaudeAgentTools]):
    pass


class ClaudeAgent(MCPAgent[BetaMessageParam, ClaudeAgentTools, ClaudeAgentState]):
    """
    Claude agent that uses MCP servers for tool execution.

    This agent uses Claude's native tool calling capabilities but executes
    tools through MCP servers instead of direct implementation.
    """

    @with_signature(ClaudeConfig)
    @classmethod
    def create(cls, **kwargs: object) -> ClaudeAgent:  # pyright: ignore[reportIncompatibleMethodOverride]
        return cls(ClaudeConfig.model_validate(kwargs))

    def __init__(self, config: ClaudeConfig | None = None) -> None:
        config = config or ClaudeConfig()
        super().__init__(config)
        self.config: ClaudeConfig

        model_client = self.config.model_client
        if model_client is None:
            # Default to HUD gateway when HUD_API_KEY is available
            if settings.api_key:
                model_client = gateway.build_gateway_client("anthropic")
            elif settings.anthropic_api_key:
                model_client = AsyncAnthropic(api_key=settings.anthropic_api_key)
            else:
                raise ValueError(
                    "No API key found for Claude.\n"
                    "  • Set HUD_API_KEY to use HUD Gateway"
                    " (add your Anthropic key at"
                    " hud.ai/project/secrets for BYOK)\n"
                    "  • Or set ANTHROPIC_API_KEY for direct"
                    " access"
                )

        self.anthropic_client: AsyncAnthropic | AsyncAnthropicBedrock = cast(
            "AsyncAnthropic | AsyncAnthropicBedrock", model_client
        )
        self.max_tokens = self.config.max_tokens

    async def initialize_state(self, prompt: list[types.PromptMessage]) -> ClaudeAgentState:
        """Format MCP prompt messages for Claude."""
        formatted: list[BetaMessageParam] = []
        for message in prompt:
            match message.content:
                case mcp_types.TextContent():
                    content = BetaTextBlockParam(type="text", text=message.content.text)
                case mcp_types.ImageContent():
                    content = BetaImageBlockParam(
                        type="image",
                        source=BetaBase64ImageSourceParam(
                            type="base64",
                            media_type=cast("ClaudeImageMediaType", message.content.mimeType),
                            data=message.content.data,
                        ),
                    )
                case mcp_types.EmbeddedResource(
                    resource=mcp_types.BlobResourceContents(mimeType="application/pdf") as resource
                ):
                    content = BetaRequestDocumentBlockParam(
                        type="document",
                        source=BetaBase64PDFSourceParam(
                            type="base64",
                            media_type="application/pdf",
                            data=resource.blob,
                        ),
                    )
                case _:
                    raise ValueError(f"Unknown content block type: {type(message.content)}")
            formatted.append(
                BetaMessageParam(
                    role=message.role,
                    content=[content],
                )
            )
        return ClaudeAgentState.model_construct(messages=formatted, tools=ClaudeAgentTools())

    async def get_response(self, state: ClaudeAgentState) -> AgentResponse:
        """Get response from Claude including any tool calls."""
        messages = state.messages
        tools = state.tools
        # Betas are collected during provider tool conversion.
        # Only pass betas when non-empty; an empty list can produce an empty
        # anthropic-beta header which the API rejects.
        betas: list[str] | Omit = list(tools.required_betas) if tools.required_betas else Omit()
        tool_choice = BetaToolChoiceAutoParam(type="auto", disable_parallel_tool_use=True)

        effective_tools: list[BetaToolUnionParam] = list(tools.params)
        if tools.tool_search_threshold is not None:
            generic_count = sum(1 for t in effective_tools if "input_schema" in t)
            if generic_count > tools.tool_search_threshold:
                logger.debug(
                    "tool_search: %d generic tools > threshold %d, applying defer_loading",
                    generic_count,
                    tools.tool_search_threshold,
                )
                effective_tools = [
                    {**t, "defer_loading": True} if "input_schema" in t else t
                    for t in effective_tools
                ]

        client = self.anthropic_client
        response: BetaMessage | None = None
        is_bedrock = isinstance(client, AsyncAnthropicBedrock)
        invalid_json_failures = 0

        for _ in range(1 if is_bedrock else 3):
            messages_cached: list[BetaMessageParam] = copy.deepcopy(messages)
            cache_control = CacheControlEphemeralParam(type="ephemeral")
            if messages_cached and messages_cached[-1].get("role") == "user":
                content = messages_cached[-1]["content"]
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block["type"] not in (
                            "redacted_thinking",
                            "thinking",
                        ):
                            cast("dict[str, object]", block)["cache_control"] = cache_control

            try:
                if isinstance(client, AsyncAnthropicBedrock):
                    response = await client.beta.messages.create(
                        model=self.config.model,
                        system=self.system_prompt if self.system_prompt is not None else Omit(),
                        max_tokens=self.max_tokens,
                        messages=messages_cached,
                        tools=effective_tools,
                        tool_choice=tool_choice,
                        betas=betas,
                    )
                else:
                    async with client.beta.messages.stream(
                        model=self.config.model,
                        system=self.system_prompt if self.system_prompt is not None else Omit(),
                        max_tokens=self.max_tokens,
                        messages=messages_cached,
                        tools=effective_tools,
                        tool_choice=tool_choice,
                        betas=betas,
                    ) as stream:
                        async for _ in stream:
                            pass
                        response = await stream.get_final_message()
                messages.append(BetaMessageParam(role="assistant", content=response.content))
                break
            except ModuleNotFoundError:
                if is_bedrock:
                    raise ValueError(
                        "boto3 is required for AWS Bedrock. Use `pip install hud-python[bedrock]`"
                    ) from None
                raise
            except ValueError as exc:
                message = str(exc)
                if is_bedrock or "Unable to parse tool parameter JSON from model." not in message:
                    raise

                marker = "JSON: "
                marker_index = message.find(marker)
                invalid_json = (
                    "" if marker_index == -1 else message[marker_index + len(marker) :].strip()
                )

                invalid_json_failures += 1
                if invalid_json_failures == 1:
                    logger.warning(
                        "Claude returned invalid streamed tool JSON; retrying same generation once"
                    )
                    continue

                if invalid_json_failures == 2:
                    wrapped = json.dumps({"INVALID_JSON": invalid_json}, ensure_ascii=True)
                    retry_text = (
                        "Your previous tool-call arguments were invalid JSON and could not be "
                        "parsed.\n"
                        "Retry the same intended tool call once with valid JSON arguments only.\n"
                        "Ensure all strings are quoted and all arrays/objects are valid JSON.\n"
                        f"Malformed payload (wrapped): {wrapped}"
                    )
                    logger.warning(
                        "Claude returned invalid streamed tool JSON twice; "
                        "retrying once with INVALID_JSON guidance"
                    )
                    messages.append(
                        BetaMessageParam(
                            role="user",
                            content=[BetaTextBlockParam(type="text", text=retry_text)],
                        )
                    )
                    continue

                raise

        if response is None:
            raise ValueError("Claude response missing after stream retries")

        result = AgentResponse(content="", tool_calls=[], done=True)
        text_content = ""
        thinking_content = ""
        citations: list[dict[str, object]] = []

        for block in response.content:
            match block.type:
                case "tool_use":
                    tool_use = block
                    mcp_name = tools.name_map.get(tool_use.name, tool_use.name)
                    result.tool_calls.append(
                        MCPToolCall(
                            id=tool_use.id,
                            name=mcp_name,
                            arguments=dict(tool_use.input),
                            _meta=mcp_types.RequestParams.Meta.model_validate(
                                {"enable_citations": self.enable_citations}
                            ),
                        )
                    )
                    result.done = False
                case "text":
                    text = cast("BetaTextBlock", block)
                    text_content += text.text
                    for citation in text.citations or []:
                        normalized = self._citation(citation)
                        citations.append(normalized.model_dump(exclude={"provider_data"}))
                case "thinking":
                    thinking = block
                    if thinking.thinking:
                        if thinking_content:
                            thinking_content += "\n"
                        thinking_content += thinking.thinking
                case _:
                    continue

        result.content = text_content
        result.citations = citations
        if thinking_content:
            result.reasoning = thinking_content

        return result

    @staticmethod
    def _citation(citation: BetaTextCitation) -> Citation:
        match citation.type:
            case "char_location":
                char_location = citation
                citation_type = "document_citation"
                text = char_location.cited_text
                source = str(char_location.document_index)
                title = char_location.document_title
                start_index = char_location.start_char_index
                end_index = char_location.end_char_index
            case "page_location":
                page_location = citation
                citation_type = "document_citation"
                text = page_location.cited_text
                source = str(page_location.document_index)
                title = page_location.document_title
                start_index = None
                end_index = None
            case "content_block_location":
                block_location = citation
                citation_type = "document_citation"
                text = block_location.cited_text
                source = str(block_location.document_index)
                title = block_location.document_title
                start_index = block_location.start_block_index
                end_index = block_location.end_block_index
            case "search_result_location":
                search_result = citation
                citation_type = "search_result_location"
                text = search_result.cited_text
                source = search_result.source
                title = search_result.title
                start_index = search_result.start_block_index
                end_index = search_result.end_block_index
            case "web_search_result_location":
                web_result = citation
                citation_type = "web_search_result_location"
                text = web_result.cited_text
                source = web_result.url
                title = web_result.title
                start_index = None
                end_index = None

        return Citation(
            type=citation_type,
            text=text,
            source=source,
            title=title,
            start_index=start_index,
            end_index=end_index,
        )
