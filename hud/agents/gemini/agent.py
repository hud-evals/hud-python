"""GeminiAgent — ``ToolAgent`` over Google's Gemini Generate Content API."""

from __future__ import annotations

import base64
import logging
from typing import Any, cast

import mcp.types as mcp_types
from google import genai
from google.genai import types as genai_types

from hud.agents import gateway
from hud.agents.tool_agent import RunState, ToolAgent
from hud.agents.types import GeminiConfig
from hud.settings import settings
from hud.tools.types import Citation
from hud.types import AgentResponse, MCPToolCall, MCPToolResult

from .settings import gemini_agent_settings
from .tools import (
    PREDEFINED_COMPUTER_USE_FUNCTIONS,
    GeminiComputerTool,
    GeminiEditTool,
    GeminiGlobTool,
    GeminiListTool,
    GeminiMCPProxyTool,
    GeminiReadTool,
    GeminiSearchTool,
    GeminiShellTool,
    GeminiWriteTool,
)

logger = logging.getLogger(__name__)


class GeminiAgent(ToolAgent[genai_types.Content]):
    """Gemini agent. Drives SSH (coding/filesystem), RFB (computer), and MCP capabilities."""

    tool_catalog = (
        GeminiShellTool,
        GeminiEditTool,
        GeminiWriteTool,
        GeminiReadTool,
        GeminiSearchTool,
        GeminiGlobTool,
        GeminiListTool,
        GeminiComputerTool,
        GeminiMCPProxyTool,
    )

    def __init__(self, config: GeminiConfig | None = None) -> None:
        config = config or GeminiConfig()
        self.config = config
        self.model = config.model
        self.auto_respond = config.auto_respond
        self.hosted_tools = list(config.hosted_tools)

        model_client = config.model_client
        if model_client is None:
            if settings.api_key:
                model_client = gateway.build_gateway_client("gemini")
            elif settings.gemini_api_key:
                model_client = genai.Client(api_key=settings.gemini_api_key)
            else:
                raise ValueError(
                    "No API key for Gemini. Set HUD_API_KEY or GEMINI_API_KEY.",
                )

        self.gemini_client: genai.Client = cast("genai.Client", model_client)
        self.temperature = config.temperature
        self.top_p = config.top_p
        self.top_k = config.top_k
        self.max_output_tokens = config.max_output_tokens
        self.thinking_level = config.thinking_level
        self.include_thoughts = config.include_thoughts
        self.excluded_predefined_functions = list(config.excluded_predefined_functions)
        self.max_recent_turn_with_screenshots = (
            gemini_agent_settings.MAX_RECENT_TURN_WITH_SCREENSHOTS
        )

    # ─── ToolAgent hooks ──────────────────────────────────────────────

    async def _initialize_state(self, *, prompt: str) -> RunState[genai_types.Content]:
        return RunState(
            messages=[
                genai_types.Content(role="user", parts=[genai_types.Part(text=prompt)]),
            ]
        )

    def _format_user_text(self, text: str) -> genai_types.Content:
        return genai_types.Content(role="user", parts=[genai_types.Part(text=text)])

    def _format_result(
        self,
        call: MCPToolCall,
        result: MCPToolResult,
    ) -> genai_types.Content | None:
        text = next(
            (c.text for c in result.content if isinstance(c, mcp_types.TextContent)),
            None,
        )
        response: dict[str, Any] = (
            {"error": text or "Tool execution failed"} if result.isError else {"success": True}
        )
        if text is not None and not result.isError:
            response["output"] = text

        parts: list[genai_types.FunctionResponsePart] = []
        for block in result.content:
            if isinstance(block, mcp_types.ImageContent):
                parts.append(
                    genai_types.FunctionResponsePart(
                        inline_data=genai_types.FunctionResponseBlob(
                            mime_type=block.mimeType or "image/png",
                            data=base64.b64decode(block.data),
                        ),
                    ),
                )

        return genai_types.Content(
            role="user",
            parts=[
                genai_types.Part(
                    function_response=genai_types.FunctionResponse(
                        name=call.provider_name or call.name,
                        response=response,
                        parts=parts or None,
                    ),
                ),
            ],
        )

    async def get_response(
        self,
        state: RunState[genai_types.Content],
        *,
        system_prompt: str | None = None,
        citations_enabled: bool = False,
    ) -> AgentResponse:
        messages = state.messages

        # Drop screenshots from older computer tool turns.
        computer_tool = self._find_computer_tool()
        predefined = frozenset(PREDEFINED_COMPUTER_USE_FUNCTIONS)
        screenshot_turns: list[list[genai_types.FunctionResponse]] = []
        for content in reversed(messages):
            if content.role != "user":
                continue
            turn_responses: list[genai_types.FunctionResponse] = []
            for part in content.parts or []:
                fr = part.function_response
                if fr is not None and fr.parts and fr.name in predefined:
                    turn_responses.append(fr)
            if turn_responses:
                screenshot_turns.append(turn_responses)
        for old_turn in screenshot_turns[self.max_recent_turn_with_screenshots :]:
            for fr in old_turn:
                fr.parts = None

        provider_tools = cast("genai_types.ToolListUnion", list(self.params))
        if citations_enabled and not any(getattr(t, "google_search", None) for t in self.params):
            provider_tools = [
                *list(provider_tools),
                genai_types.Tool(google_search=genai_types.GoogleSearch()),
            ]

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
            tools=provider_tools,
            system_instruction=system_prompt,
            thinking_config=thinking_config,
        )

        api_response = await self.gemini_client.aio.models.generate_content(
            model=self.model,
            contents=cast("Any", messages),
            config=generate_config,
        )
        if not api_response.candidates:
            raise RuntimeError(f"Gemini returned no candidates for model {self.model}")

        candidate = api_response.candidates[0]
        content = candidate.content
        if content is not None:
            messages.append(content)

        result = AgentResponse(content="", tool_calls=[], done=True)
        text_parts: list[str] = []
        thought_parts: list[str] = []

        for part in (content.parts or []) if content else []:
            function_call = part.function_call
            if function_call is not None:
                tc = self._make_tool_call(function_call, computer_tool)
                result.tool_calls.append(tc)
                result.done = False
                continue
            if part.text:
                if part.thought is True:
                    thought_parts.append(part.text)
                else:
                    text_parts.append(part.text)

        result.content = "".join(text_parts)
        if thought_parts:
            result.reasoning = "\n".join(thought_parts)

        grounding_meta = candidate.grounding_metadata
        if grounding_meta is not None:
            result.citations = [
                c.model_dump(exclude={"provider_data"})
                for c in _grounding_citations(grounding_meta)
            ]

        return result

    def _find_computer_tool(self) -> GeminiComputerTool | None:
        for tool in self.tools.values():
            if isinstance(tool, GeminiComputerTool):
                return tool
        return None

    def _make_tool_call(
        self,
        function_call: genai_types.FunctionCall,
        computer_tool: GeminiComputerTool | None,
    ) -> MCPToolCall:
        name = function_call.name or ""
        arguments = dict(function_call.args) if function_call.args else {}
        predefined = frozenset(PREDEFINED_COMPUTER_USE_FUNCTIONS)
        if computer_tool is not None and name in predefined:
            return MCPToolCall(
                name=computer_tool.name,
                arguments={"action": name, **arguments},
                provider_name=name,
            )
        return MCPToolCall(name=name, arguments=arguments)


def _grounding_citations(grounding_meta: genai_types.GroundingMetadata) -> list[Citation]:
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


__all__ = ["GeminiAgent"]
