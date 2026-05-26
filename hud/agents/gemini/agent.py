"""Gemini MCP Agent implementation."""

from __future__ import annotations

import base64
import logging
from functools import cached_property
from typing import Any, cast

import mcp.types as types
from google import genai
from google.genai import types as genai_types

from hud.agents import gateway
from hud.agents.base import MCPAgent
from hud.agents.types import GeminiConfig
from hud.settings import settings
from hud.tools.types import Citation
from hud.types import AgentResponse
from hud.utils.types import with_signature

from .settings import gemini_agent_settings
from .tools import GeminiAgentTools

logger = logging.getLogger(__name__)


class GeminiAgent(MCPAgent[genai_types.Content]):
    """
    Gemini agent that uses MCP servers for tool execution.

    This agent uses Gemini's native tool calling capabilities but executes
    tools through MCP servers instead of direct implementation.
    """

    @with_signature(GeminiConfig)
    @classmethod
    def create(cls, **kwargs: object) -> GeminiAgent:  # pyright: ignore[reportIncompatibleMethodOverride]
        return cls(GeminiConfig.model_validate(kwargs))

    def __init__(self, config: GeminiConfig | None = None) -> None:
        config = config or GeminiConfig()
        super().__init__(config)
        self.config: GeminiConfig

        model_client = self.config.model_client
        if model_client is None:
            if settings.api_key:
                model_client = gateway.build_gateway_client("gemini")
            elif settings.gemini_api_key:
                model_client = genai.Client(api_key=settings.gemini_api_key)
                if self.config.validate_api_key:
                    try:
                        next(iter(model_client.models.list()), None)
                    except Exception as e:
                        raise ValueError(f"Gemini API key is invalid: {e}") from e
            else:
                raise ValueError(
                    "No API key found for Gemini.\n"
                    "  • Set HUD_API_KEY to use HUD Gateway"
                    " (add your Gemini key at"
                    " hud.ai/project/secrets for BYOK)\n"
                    "  • Or set GEMINI_API_KEY for direct"
                    " access"
                )

        self.gemini_client: genai.Client = cast("genai.Client", model_client)
        self.temperature = self.config.temperature
        self.top_p = self.config.top_p
        self.top_k = self.config.top_k
        self.max_output_tokens = self.config.max_output_tokens
        self.thinking_level = self.config.thinking_level
        self.include_thoughts = self.config.include_thoughts

        self.excluded_predefined_functions = list(self.config.excluded_predefined_functions)
        self.max_recent_turn_with_screenshots = (
            gemini_agent_settings.MAX_RECENT_TURN_WITH_SCREENSHOTS
        )

    @cached_property
    def tools(self) -> GeminiAgentTools:
        return GeminiAgentTools(
            excluded_predefined_functions=self.excluded_predefined_functions,
        )

    async def format_messages(
        self, messages: list[types.PromptMessage]
    ) -> list[genai_types.Content]:
        """Format MCP prompt messages for Gemini."""
        return [
            genai_types.Content(
                role="model" if str(message.role) == "assistant" else str(message.role),
                parts=[_format_content(message.content)],
            )
            for message in messages
        ]

    async def get_response(self, messages: list[genai_types.Content]) -> AgentResponse:
        """Get response from Gemini including any tool calls."""
        # Drop screenshots from older computer tool responses to keep context small.
        screenshot_turns: list[list[genai_types.FunctionResponse]] = []
        for content in reversed(messages):
            if content.role != "user":
                continue

            turn_responses: list[genai_types.FunctionResponse] = []
            for part in content.parts or []:
                function_response = part.function_response
                if (
                    function_response is not None
                    and function_response.parts
                    and function_response.name in self.tools.predefined_computer_functions
                ):
                    turn_responses.append(function_response)

            if turn_responses:
                screenshot_turns.append(turn_responses)

        for old_turn in screenshot_turns[self.max_recent_turn_with_screenshots :]:
            for function_response in old_turn:
                function_response.parts = None

        # Configure Gemini generation options.
        tools = cast("genai_types.ToolListUnion", self.tools.params)
        if self.enable_citations and not any(tool.google_search for tool in self.tools.params):
            tools = [*list(tools), genai_types.Tool(google_search=genai_types.GoogleSearch())]

        thinking_config = None
        if self.thinking_level is not None or self.include_thoughts:
            thinking_config = genai_types.ThinkingConfig(
                thinking_level=genai_types.ThinkingLevel(self.thinking_level.upper())
                if self.thinking_level is not None
                else None,
                include_thoughts=self.include_thoughts,
            )

        generate_config = genai_types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_output_tokens=self.max_output_tokens,
            tools=tools,
            system_instruction=self.system_prompt,
            thinking_config=thinking_config,
        )

        api_response = await self.gemini_client.aio.models.generate_content(
            model=self.config.model,
            contents=cast("Any", messages),
            config=generate_config,
        )
        if not api_response.candidates:
            detail_parts: list[str] = []
            if api_response.prompt_feedback is not None:
                detail_parts.append(
                    f"prompt_feedback={api_response.prompt_feedback.model_dump_json()}"
                )
            if api_response.usage_metadata is not None:
                detail_parts.append(
                    f"usage_metadata={api_response.usage_metadata.model_dump_json()}"
                )
            details = "; ".join(detail_parts) if detail_parts else "no response metadata"
            raise RuntimeError(
                f"Gemini response returned no candidates for model {self.config.model}. {details}"
            )

        candidate = api_response.candidates[0]

        # Append assistant response (including any function_call) so that
        # subsequent FunctionResponse messages correspond to a prior FunctionCall
        content = candidate.content
        if content is not None:
            messages.append(content)

        # Normalize text, thoughts, tool calls, and citations.
        result = AgentResponse(content="", tool_calls=[], done=True)
        text_parts: list[str] = []
        thought_parts: list[str] = []

        parts = []
        if content is not None:
            parts = content.parts or []
        for part in parts:
            function_call = part.function_call
            if function_call is not None:
                result.tool_calls.append(self.tools.tool_call(function_call))
                result.done = False
                continue

            if not part.text:
                continue

            if part.thought is True:
                thought_parts.append(part.text)
            else:
                text_parts.append(part.text)

        result.content = "".join(text_parts)
        if thought_parts:
            result.reasoning = "\n".join(thought_parts)

        grounding_meta = candidate.grounding_metadata
        if grounding_meta is not None:
            # TODO: Also normalize candidate.citation_metadata for URL-context citation spans.
            result.citations = [
                citation.model_dump(exclude={"provider_data"})
                for citation in _grounding_citations(grounding_meta)
            ]

        return result


def _format_content(
    content: types.ContentBlock,
) -> genai_types.Part:
    match content:
        case types.TextContent(text=text):
            return genai_types.Part(text=text)
        case types.ImageContent(data=data, mimeType=mime_type):
            return genai_types.Part.from_bytes(
                data=base64.b64decode(data),
                mime_type=mime_type or "image/png",
            )
        case _:
            raise ValueError(f"Unknown content block type: {type(content)}")


def _grounding_citations(
    grounding_meta: genai_types.GroundingMetadata,
) -> list[Citation]:
    citations: list[Citation] = []
    chunk_sources: list[tuple[str, str | None]] = []
    for chunk in grounding_meta.grounding_chunks or []:
        if chunk.web is None:
            chunk_sources.append(("", None))
        else:
            chunk_sources.append((chunk.web.uri or "", chunk.web.title))

    seen_chunk_indices: set[int] = set()
    for support in grounding_meta.grounding_supports or []:
        segment = support.segment
        segment_text = segment.text or "" if segment else ""
        start_idx = segment.start_index if segment else None
        end_idx = segment.end_index if segment else None

        for idx in support.grounding_chunk_indices or []:
            seen_chunk_indices.add(idx)
            source, title = chunk_sources[idx] if 0 <= idx < len(chunk_sources) else ("", None)
            citations.append(
                Citation(
                    type="grounding",
                    text=segment_text,
                    source=source,
                    title=title,
                    start_index=start_idx,
                    end_index=end_idx,
                )
            )

    for idx, (source, title) in enumerate(chunk_sources):
        if idx not in seen_chunk_indices and source:
            citations.append(Citation(type="grounding", text="", source=source, title=title))
    return citations
