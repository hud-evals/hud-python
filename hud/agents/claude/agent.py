"""ClaudeAgent — ``ToolAgent`` over Anthropic's Messages API."""

from __future__ import annotations

import copy
import json
import logging
from typing import TYPE_CHECKING, Any, Literal, cast

import mcp.types as mcp_types
from anthropic import AsyncAnthropic, AsyncAnthropicBedrock, Omit
from anthropic.types import CacheControlEphemeralParam
from anthropic.types.beta import (
    BetaBase64ImageSourceParam,
    BetaBase64PDFSourceParam,
    BetaImageBlockParam,
    BetaMessage,
    BetaMessageParam,
    BetaPlainTextSourceParam,
    BetaRequestDocumentBlockParam,
    BetaTextBlockParam,
    BetaToolChoiceAutoParam,
    BetaToolResultBlockParam,
    BetaToolUnionParam,
)

from hud.agents.tool_agent import RunState, ToolAgent
from hud.agents.types import Citation, ClaudeConfig
from hud.settings import settings
from hud.types import AgentResponse, MCPToolCall, MCPToolResult
from hud.utils import gateway

from .tools.coding import ClaudeBashTool, ClaudeTextEditorTool
from .tools.computer import ClaudeComputerTool
from .tools.mcp_proxy import ClaudeMCPProxyTool

if TYPE_CHECKING:
    from anthropic.types.beta import BetaTextBlock, BetaTextCitation

logger = logging.getLogger(__name__)

ClaudeImageMediaType = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
ClaudeToolResultContent = BetaTextBlockParam | BetaImageBlockParam | BetaRequestDocumentBlockParam


class ClaudeAgent(ToolAgent[BetaMessageParam, ClaudeConfig]):
    """Anthropic Claude agent. Drives SSH (coding), RFB (computer), and MCP capabilities."""

    tool_catalog = (
        ClaudeBashTool,
        ClaudeTextEditorTool,
        ClaudeComputerTool,
        ClaudeMCPProxyTool,
    )

    def __init__(self, config: ClaudeConfig | None = None) -> None:
        self.config = config or ClaudeConfig()
        self.anthropic_client: AsyncAnthropic | AsyncAnthropicBedrock = self._resolve_client()

    def _resolve_client(self) -> AsyncAnthropic | AsyncAnthropicBedrock:
        if self.config.model_client is not None:
            return cast("AsyncAnthropic | AsyncAnthropicBedrock", self.config.model_client)
        if settings.api_key:
            return cast("AsyncAnthropic", gateway.build_gateway_client("anthropic"))
        if settings.anthropic_api_key:
            return AsyncAnthropic(api_key=settings.anthropic_api_key)
        raise ValueError(
            "No API key found for Claude. Set HUD_API_KEY (gateway) or ANTHROPIC_API_KEY.",
        )

    # ─── ToolAgent hooks ──────────────────────────────────────────────

    async def _initialize_state(
        self, *, prompt: str | list[Any] | None
    ) -> RunState[BetaMessageParam]:
        return RunState(messages=self._initial_messages(prompt))

    def _format_message(self, role: str, text: str) -> BetaMessageParam:
        return BetaMessageParam(
            role="assistant" if role == "assistant" else "user",
            content=[BetaTextBlockParam(type="text", text=text)],
        )

    def _format_result(
        self,
        call: MCPToolCall,
        result: MCPToolResult,
        state: RunState[BetaMessageParam],
    ) -> BetaMessageParam | list[BetaMessageParam] | None:
        tool_use_id = call.id
        if not tool_use_id:
            return None

        result_content = result.content
        if result.isError:
            error_msg = next(
                (c.text for c in result.content if isinstance(c, mcp_types.TextContent)),
                "Tool execution failed",
            )
            result_content = [mcp_types.TextContent(type="text", text=f"Error: {error_msg}")]

        citations_enabled = bool(getattr(call.meta, "citations_enabled", False))
        claude_blocks: list[ClaudeToolResultContent] = []
        sibling_docs: list[BetaRequestDocumentBlockParam] = []

        for content in result_content:
            citation_doc: BetaRequestDocumentBlockParam | None = None
            match content:
                case mcp_types.TextContent():
                    block = BetaTextBlockParam(type="text", text=content.text)
                    if citations_enabled and not result.isError:
                        citation_doc = BetaRequestDocumentBlockParam(
                            type="document",
                            source=BetaPlainTextSourceParam(
                                type="text",
                                media_type="text/plain",
                                data=content.text,
                            ),
                            title=call.name,
                            citations={"enabled": True},
                        )
                case mcp_types.ImageContent():
                    block = BetaImageBlockParam(
                        type="image",
                        source=BetaBase64ImageSourceParam(
                            type="base64",
                            media_type=cast("ClaudeImageMediaType", content.mimeType),
                            data=content.data,
                        ),
                    )
                case mcp_types.EmbeddedResource(
                    resource=mcp_types.BlobResourceContents(mimeType="application/pdf") as resource,
                ):
                    block = BetaRequestDocumentBlockParam(
                        type="document",
                        source=BetaBase64PDFSourceParam(
                            type="base64",
                            media_type="application/pdf",
                            data=resource.blob,
                        ),
                    )
                    if citations_enabled and not result.isError:
                        citation_doc = BetaRequestDocumentBlockParam(
                            type="document",
                            source=block["source"],
                            citations={"enabled": True},
                        )
                case _:
                    raise ValueError(f"Unknown content block type: {type(content)}")

            claude_blocks.append(block)
            if citation_doc is not None:
                sibling_docs.append(citation_doc)

        tool_result_msg = BetaMessageParam(
            role="user",
            content=[
                BetaToolResultBlockParam(
                    type="tool_result",
                    tool_use_id=tool_use_id,
                    content=claude_blocks,
                ),
            ],
        )
        if sibling_docs:
            return [tool_result_msg, BetaMessageParam(role="user", content=sibling_docs)]
        return tool_result_msg

    # ─── Anthropic call ───────────────────────────────────────────────

    async def get_response(
        self,
        state: RunState[BetaMessageParam],
        *,
        system_prompt: str | None = None,
        citations_enabled: bool = False,
    ) -> AgentResponse:
        required_betas = {
            beta for tool in state.tools.values() if (beta := getattr(tool.spec, "beta", None))
        }
        betas: list[str] | Omit = list(required_betas) if required_betas else Omit()
        tool_choice = BetaToolChoiceAutoParam(type="auto", disable_parallel_tool_use=True)
        tools = cast("list[BetaToolUnionParam]", list(state.params))
        system = system_prompt if system_prompt is not None else Omit()
        is_bedrock = isinstance(self.anthropic_client, AsyncAnthropicBedrock)

        response: BetaMessage | None = None
        invalid_json_failures = 0

        for _ in range(1 if is_bedrock else 3):
            messages_cached = self._cache_last_user_block(copy.deepcopy(state.messages))
            try:
                if is_bedrock:
                    response = await self.anthropic_client.beta.messages.create(
                        model=self.config.model,
                        system=system,
                        max_tokens=self.config.max_tokens,
                        messages=messages_cached,
                        tools=tools,
                        tool_choice=tool_choice,
                        betas=betas,
                    )
                else:
                    client = cast("AsyncAnthropic", self.anthropic_client)
                    async with client.beta.messages.stream(
                        model=self.config.model,
                        system=system,
                        max_tokens=self.config.max_tokens,
                        messages=messages_cached,
                        tools=tools,
                        tool_choice=tool_choice,
                        betas=betas,
                    ) as stream:
                        async for _ in stream:
                            pass
                        response = await stream.get_final_message()

                state.messages.append(
                    BetaMessageParam(role="assistant", content=response.content),
                )
                break

            except ValueError as exc:
                message = str(exc)
                if is_bedrock or "Unable to parse tool parameter JSON from model." not in message:
                    raise

                invalid_json_failures += 1
                if invalid_json_failures == 1:
                    logger.warning("Claude returned invalid tool JSON; retrying once")
                    continue

                if invalid_json_failures == 2:
                    marker = "JSON: "
                    idx = message.find(marker)
                    payload = "" if idx == -1 else message[idx + len(marker) :].strip()
                    wrapped = json.dumps({"INVALID_JSON": payload}, ensure_ascii=True)
                    state.messages.append(
                        BetaMessageParam(
                            role="user",
                            content=[
                                BetaTextBlockParam(
                                    type="text",
                                    text=(
                                        "Your previous tool-call arguments were invalid JSON. "
                                        "Retry the same tool call with valid JSON arguments.\n"
                                        f"Malformed payload (wrapped): {wrapped}"
                                    ),
                                )
                            ],
                        )
                    )
                    continue

                raise

        if response is None:
            raise ValueError("Claude response missing after retries")

        result = AgentResponse(content="", tool_calls=[], done=True)
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        citations: list[dict[str, object]] = []

        for block in response.content:
            match block.type:
                case "tool_use":
                    arguments = dict(block.input) if block.input else {}
                    result.tool_calls.append(
                        MCPToolCall(
                            id=block.id,
                            name=block.name,
                            arguments=arguments,
                            _meta=mcp_types.RequestParams.Meta.model_validate(
                                {"citations_enabled": citations_enabled},
                            ),
                        )
                    )
                    result.done = False
                case "text":
                    text_block = cast("BetaTextBlock", block)
                    text_parts.append(text_block.text)
                    citations.extend(
                        self._citation(c).model_dump(exclude={"provider_data"})
                        for c in (text_block.citations or [])
                    )
                case "thinking":
                    if block.thinking:
                        thinking_parts.append(block.thinking)
                case _:
                    pass

        result.content = "".join(text_parts)
        result.citations = citations
        if thinking_parts:
            result.reasoning = "\n".join(thinking_parts)
        result.finish_reason = response.stop_reason
        return result

    @staticmethod
    def _cache_last_user_block(
        messages: list[BetaMessageParam],
    ) -> list[BetaMessageParam]:
        if not messages or messages[-1].get("role") != "user":
            return messages
        content = messages[-1]["content"]
        if not isinstance(content, list):
            return messages
        cache_control = CacheControlEphemeralParam(type="ephemeral")
        skip = {"redacted_thinking", "thinking"}
        for block in content:
            if isinstance(block, dict) and block.get("type") not in skip:
                cast("dict[str, object]", block)["cache_control"] = cache_control
        return messages

    @staticmethod
    def _citation(citation: BetaTextCitation) -> Citation:
        match citation.type:
            case "char_location":
                return Citation(
                    type="document_citation",
                    text=citation.cited_text,
                    source=str(citation.document_index),
                    title=citation.document_title,
                    start_index=citation.start_char_index,
                    end_index=citation.end_char_index,
                )
            case "page_location":
                return Citation(
                    type="document_citation",
                    text=citation.cited_text,
                    source=str(citation.document_index),
                    title=citation.document_title,
                    start_index=None,
                    end_index=None,
                )
            case "content_block_location":
                return Citation(
                    type="document_citation",
                    text=citation.cited_text,
                    source=str(citation.document_index),
                    title=citation.document_title,
                    start_index=citation.start_block_index,
                    end_index=citation.end_block_index,
                )
            case "search_result_location":
                return Citation(
                    type="search_result_location",
                    text=citation.cited_text,
                    source=citation.source,
                    title=citation.title,
                    start_index=citation.start_block_index,
                    end_index=citation.end_block_index,
                )
            case "web_search_result_location":
                return Citation(
                    type="web_search_result_location",
                    text=citation.cited_text,
                    source=citation.url,
                    title=citation.title,
                    start_index=None,
                    end_index=None,
                )


__all__ = ["ClaudeAgent"]
