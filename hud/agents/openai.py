"""OpenAI MCP Agent implementation."""

from __future__ import annotations

import copy
import json
import logging
from inspect import cleandoc
from typing import Any, ClassVar, Literal

import mcp.types as types
from openai import AsyncOpenAI, Omit, OpenAI
from openai.types.responses import (
    ApplyPatchToolParam,
    ComputerToolParam,
    ComputerUsePreviewToolParam,
    FunctionShellToolParam,
    FunctionToolParam,
    ResponseComputerToolCallOutputScreenshotParam,
    ResponseFunctionCallOutputItemListParam,
    ResponseIncludable,
    ResponseInputFileContentParam,
    ResponseInputImageContentParam,
    ResponseInputImageParam,
    ResponseInputMessageContentListParam,
    ResponseInputParam,
    ResponseInputTextContentParam,
    ResponseInputTextParam,
    ResponseOutputText,
    ToolParam,
)
from openai.types.responses.response_create_params import ToolChoice  # noqa: TC002
from openai.types.responses.response_input_param import (
    ComputerCallOutput,
    ComputerCallOutputAcknowledgedSafetyCheck,
    FunctionCallOutput,
    Message,
)
from openai.types.shared_params.reasoning import Reasoning  # noqa: TC002

from hud.settings import settings
from hud.tools.native_types import NativeToolSpec
from hud.types import AgentType, BaseAgentConfig, InferenceResult, MCPToolCall, MCPToolResult, Trace
from hud.utils.strict_schema import ensure_strict_json_schema
from hud.utils.types import with_signature

from .base import MCPAgent
from .types import OpenAIConfig, OpenAICreateParams

logger = logging.getLogger(__name__)


class OpenAIAgent(MCPAgent):
    """Generic OpenAI agent that can execute MCP tools through the Responses API."""

    metadata: ClassVar[dict[str, Any] | None] = None
    config_cls: ClassVar[type[BaseAgentConfig]] = OpenAIConfig

    @classmethod
    def agent_type(cls) -> AgentType:
        """Return the AgentType for OpenAI."""
        return AgentType.OPENAI

    # Legacy tool name patterns for backwards compatibility
    _LEGACY_SHELL_NAMES = ("shell",)
    _LEGACY_APPLY_PATCH_NAMES = ("apply_patch",)

    def _legacy_native_spec_fallback(self, tool: types.Tool) -> NativeToolSpec | None:
        """Detect OpenAI native tools by name for backwards compatibility.

        Supports old environments that expose tools like 'shell' or 'apply_patch'
        without native_tools metadata.

        Each tuple is ordered by preference — first name that exists wins.
        Only returns a spec if this tool IS that preferred match.
        """
        available = {t.name for t in (self._available_tools or [])} | {tool.name}
        preferred = lambda names: next((n for n in names if n in available), None) == tool.name

        if preferred(self._LEGACY_SHELL_NAMES):
            logger.debug("Legacy fallback: detected %s as shell tool", tool.name)
            return NativeToolSpec(
                api_type="shell",
                api_name="shell",
                role="shell",
            )

        if preferred(self._LEGACY_APPLY_PATCH_NAMES):
            logger.debug("Legacy fallback: detected %s as apply_patch tool", tool.name)
            return NativeToolSpec(
                api_type="apply_patch",
                api_name="apply_patch",
                role="editor",
            )

        return None

    @with_signature(OpenAICreateParams)
    @classmethod
    def create(cls, **kwargs: Any) -> OpenAIAgent:  # pyright: ignore[reportIncompatibleMethodOverride]
        return MCPAgent.create.__func__(cls, **kwargs)  # type: ignore[return-value]

    def __init__(self, params: OpenAICreateParams | None = None, **kwargs: Any) -> None:
        super().__init__(params, **kwargs)
        self.config: OpenAIConfig

        model_client = self.config.model_client
        if model_client is None:
            if settings.api_key:
                from hud.agents.gateway import build_gateway_client

                model_client = build_gateway_client("openai")
            elif settings.openai_api_key:
                model_client = AsyncOpenAI(api_key=settings.openai_api_key)
                if self.config.validate_api_key:
                    try:
                        OpenAI(api_key=settings.openai_api_key).models.list()
                    except Exception as exc:  # pragma: no cover - network validation
                        raise ValueError(f"OpenAI API key is invalid: {exc}") from exc
            else:
                raise ValueError(
                    "No API key found for OpenAI.\n"
                    "  • Set HUD_API_KEY to use HUD Gateway"
                    " (add your OpenAI key at"
                    " hud.ai/project/secrets for BYOK)\n"
                    "  • Or set OPENAI_API_KEY for direct"
                    " access"
                )

        self.openai_client: AsyncOpenAI = model_client
        self._model = self.config.model
        self.max_output_tokens = self.config.max_output_tokens
        self.temperature = self.config.temperature
        self.reasoning: Reasoning | None = self.config.reasoning
        self.tool_choice: ToolChoice | None = self.config.tool_choice
        self.parallel_tool_calls = self.config.parallel_tool_calls
        self.text = self.config.text
        self.truncation: Literal["auto", "disabled"] | None = self.config.truncation

        self._openai_tools: list[ToolParam] = []
        self._tool_name_map: dict[str, str] = {}
        self._tool_search_threshold: int | None = None

        self.last_response_id: str | None = None
        self._message_cursor = 0
        self.pending_call_id: str | None = None
        self.pending_safety_checks: list[Any] = []

    def _on_tools_ready(self) -> None:
        """Build OpenAI-specific tool mappings after tools are discovered."""
        self._convert_tools_for_openai()

    def _build_native_tool(self, tool: types.Tool, spec: NativeToolSpec) -> ToolParam | None:
        """Build an OpenAI native tool from a NativeToolSpec.

        Args:
            tool: The MCP tool
            spec: The native spec for OpenAI

        Returns:
            OpenAI-specific tool parameter
        """
        match spec.api_type:
            case "shell":
                return FunctionShellToolParam(type="shell")
            case "apply_patch":
                return ApplyPatchToolParam(type="apply_patch")
            case "computer":
                return ComputerToolParam(type="computer")
            case "computer_use_preview":
                return ComputerUsePreviewToolParam(
                    type="computer_use_preview",
                    display_width=int(spec.extra.get("display_width", 1024)),
                    display_height=int(spec.extra.get("display_height", 768)),
                    environment=spec.extra.get("environment", "browser"),
                )
            case _:
                logger.warning(
                    "Unknown native tool type %s for tool %s, using function format",
                    spec.api_type,
                    tool.name,
                )
                return self._to_function_tool(tool)

    def _to_function_tool(self, tool: types.Tool) -> FunctionToolParam | None:
        """Convert an MCP tool to OpenAI function tool format.

        Args:
            tool: MCP tool to convert

        Returns:
            OpenAI function tool parameter
        """
        if tool.description is None or tool.inputSchema is None:
            raise ValueError(
                cleandoc(f"""MCP tool {tool.name} requires both a description and inputSchema.
                Add these by:
                1. Adding a docstring to your @mcp.tool decorated function for the description
                2. Using pydantic Field() annotations on function parameters for the schema
                """)
            )

        try:
            strict_schema = ensure_strict_json_schema(copy.deepcopy(tool.inputSchema))
        except Exception as e:
            self.console.warning_log(f"Failed to convert tool '{tool.name}' schema to strict: {e}")
            return None

        return FunctionToolParam(
            type="function",
            name=tool.name,
            description=tool.description,
            parameters=strict_schema,
            strict=True,
        )

    def _convert_tools_for_openai(self) -> None:
        """Convert MCP tools into OpenAI Responses tool definitions.

        Uses shared categorize_tools() for role-based exclusion.
        """
        self._openai_tools = []
        self._tool_name_map = {}
        self._tool_search_threshold = None

        categorized = self._categorized_tools

        # Process hosted tools
        for tool, spec in categorized.hosted:
            if not spec.api_type:
                logger.debug("Skipping hosted tool %s: no api_type", tool.name)
                continue
            tool_def: dict[str, Any] = {"type": spec.api_type}
            api_extra = {k: v for k, v in spec.extra.items() if k != "threshold"}
            tool_def.update(api_extra)
            if "threshold" in spec.extra:
                self._tool_search_threshold = spec.extra["threshold"]
            # Validate required config before sending to API
            if spec.api_type == "code_interpreter" and "container" not in spec.extra:
                raise ValueError(
                    f"Tool '{tool.name}' requires container configuration for OpenAI. "
                    "Use: CodeExecutionTool(container={'image': 'python:3.12'})"
                )
            self._openai_tools.append(tool_def)  # type: ignore[arg-type]
            logger.debug("Added hosted tool %s (%s) for OpenAI", tool.name, spec.api_type)

        # Process native tools
        for tool, spec in categorized.native:
            openai_tool = self._build_native_tool(tool, spec)
            if openai_tool:
                # Map the API name to MCP tool name for routing responses
                api_name = spec.api_name or tool.name
                self._tool_name_map[api_name] = tool.name
                self._openai_tools.append(openai_tool)

        # Process generic tools (function tools)
        for tool in categorized.generic:
            openai_tool = self._to_function_tool(tool)
            if openai_tool:
                self._tool_name_map[tool.name] = tool.name
                self._openai_tools.append(openai_tool)

        # Log actual tools being used
        tool_names = sorted(self._tool_name_map.keys())
        self.console.info(
            f"Agent initialized with {len(tool_names)} tools: {', '.join(tool_names)}"
        )

    def _extract_tool_call(self, item: Any) -> MCPToolCall | None:
        """Extract an MCPToolCall from a response output item.

        Subclasses can override to customize tool call extraction (e.g., routing
        computer_call to a different tool name).
        """
        if item.type == "function_call":
            tool_name = item.name or ""
            target_name = self._tool_name_map.get(tool_name, tool_name)
            arguments = json.loads(item.arguments)
            return MCPToolCall(name=target_name, arguments=arguments, id=item.call_id)
        elif item.type == "computer_call":
            self.pending_safety_checks = item.pending_safety_checks or []
            target_name = self._tool_name_map.get("computer", "openai_computer")
            if hasattr(item, "actions") and item.actions:
                arguments = {"actions": [a.to_dict() for a in item.actions]}
            else:
                arguments = item.action.to_dict()
            return MCPToolCall(name=target_name, arguments=arguments, id=item.call_id)
        elif item.type == "shell_call":
            target_name = self._tool_name_map.get("shell", "shell")
            return MCPToolCall(name=target_name, arguments=item.action.to_dict(), id=item.call_id)
        elif item.type == "apply_patch_call":
            target_name = self._tool_name_map.get("apply_patch", "apply_patch")
            return MCPToolCall(
                name=target_name, arguments=item.operation.to_dict(), id=item.call_id
            )
        return None

    async def _run_context(
        self, context: list[types.ContentBlock], *, max_steps: int = 10
    ) -> Trace:
        """Reset internal state before delegating to the base loop."""
        self._reset_response_state()
        return await super()._run_context(context, max_steps=max_steps)

    def _reset_response_state(self) -> None:
        self.last_response_id = None
        self._message_cursor = 0
        self.pending_call_id = None
        self.pending_safety_checks = []

    async def get_system_messages(self) -> list[types.ContentBlock]:
        """System messages are provided via the `instructions` field."""
        return []

    async def format_blocks(self, blocks: list[types.ContentBlock]) -> list[Message]:
        """Convert MCP content blocks into OpenAI user messages."""
        content: ResponseInputMessageContentListParam = []
        for block in blocks:
            if isinstance(block, types.TextContent):
                content.append(ResponseInputTextParam(type="input_text", text=block.text))
            elif isinstance(block, types.ImageContent):
                mime_type = getattr(block, "mimeType", "image/png")
                content.append(
                    ResponseInputImageParam(
                        type="input_image",
                        image_url=f"data:{mime_type};base64,{block.data}",
                        detail="auto",
                    )
                )
        if not content:
            content.append(ResponseInputTextParam(type="input_text", text=""))
        return [Message(role="user", content=content)]

    async def get_response(self, messages: ResponseInputParam) -> InferenceResult:
        """Send the latest input items to OpenAI's Responses API."""
        new_items: ResponseInputParam = messages[self._message_cursor :]
        if not new_items:
            if self.last_response_id is None:
                new_items = [
                    Message(
                        role="user", content=[ResponseInputTextParam(type="input_text", text="")]
                    )
                ]
            else:
                self.console.debug("No new messages to send to OpenAI.")
                return InferenceResult(content="", tool_calls=[], done=True)

        scenario_enable_citations = bool(
            getattr(self.ctx, "scenario_enable_citations", False) if self.ctx is not None else False
        )
        include_param: list[ResponseIncludable] | Omit = Omit()
        if scenario_enable_citations:
            include_param = ["web_search_call.action.sources"]

        effective_tools: list[ToolParam] = list(self._openai_tools)
        if self._tool_search_threshold is not None:
            fn_count = sum(
                1 for t in effective_tools if isinstance(t, dict) and t.get("type") == "function"
            )
            if fn_count > self._tool_search_threshold:
                logger.debug(
                    "tool_search: %d function tools > threshold %d, applying defer_loading",
                    fn_count,
                    self._tool_search_threshold,
                )
                effective_tools = [  # type: ignore[assignment]
                    {**t, "defer_loading": True}
                    if isinstance(t, dict) and t.get("type") == "function"
                    else t
                    for t in effective_tools
                ]

        response = await self.openai_client.responses.create(
            model=self._model,
            input=new_items,
            instructions=self.system_prompt,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            text=self.text if self.text is not None else Omit(),
            tool_choice=self.tool_choice if self.tool_choice is not None else Omit(),
            parallel_tool_calls=self.parallel_tool_calls,
            reasoning=self.reasoning if self.reasoning is not None else Omit(),
            tools=effective_tools if effective_tools else Omit(),
            previous_response_id=(
                self.last_response_id if self.last_response_id is not None else Omit()
            ),
            truncation=self.truncation if self.truncation is not None else Omit(),
            include=include_param,
        )

        self.last_response_id = response.id
        self._message_cursor = len(messages)

        agent_response = InferenceResult(content="", tool_calls=[], done=True)
        text_chunks: list[str] = []
        reasoning_chunks: list[str] = []

        citations: list[dict[str, Any]] = []

        for item in response.output:
            if item.type == "message":
                for content_block in item.content:
                    if isinstance(content_block, ResponseOutputText):
                        if content_block.text:
                            text_chunks.append(content_block.text)
                        # Extract citations from annotations
                        if content_block.annotations:
                            for ann in content_block.annotations:
                                ann_type = getattr(ann, "type", "")
                                if ann_type == "url_citation":
                                    cit_obj = getattr(ann, "url_citation", ann)
                                    citations.append(
                                        {
                                            "type": "url_citation",
                                            "text": getattr(cit_obj, "title", "") or "",
                                            "source": getattr(cit_obj, "url", "") or "",
                                            "title": getattr(cit_obj, "title", None),
                                            "start_index": getattr(ann, "start_index", None),
                                            "end_index": getattr(ann, "end_index", None),
                                        }
                                    )
                                elif ann_type == "file_citation":
                                    cit_obj = getattr(ann, "file_citation", ann)
                                    citations.append(
                                        {
                                            "type": "file_citation",
                                            "text": getattr(cit_obj, "filename", "") or "",
                                            "source": getattr(cit_obj, "file_id", "") or "",
                                            "title": getattr(cit_obj, "filename", None),
                                            "start_index": getattr(ann, "start_index", None),
                                            "end_index": getattr(ann, "end_index", None),
                                        }
                                    )
            elif item.type == "reasoning":
                reasoning_chunks.append("".join(summary.text for summary in item.summary))
            else:
                tool_call = self._extract_tool_call(item)
                if tool_call is not None:
                    agent_response.tool_calls.append(tool_call)

        if agent_response.tool_calls:
            agent_response.done = False

        agent_response.content = "".join(text_chunks)
        agent_response.citations = citations
        if reasoning_chunks:
            agent_response.reasoning = "\n".join(reasoning_chunks)
        return agent_response

    async def format_tool_results(
        self, tool_calls: list[MCPToolCall], tool_results: list[MCPToolResult]
    ) -> list[ComputerCallOutput | FunctionCallOutput]:
        """Convert MCP tool outputs into Responses input items.

        Detects computer tool results and formats them as ComputerCallOutput
        with screenshots. Non-computer calls are formatted as FunctionCallOutput.
        """
        computer_tool_name = self._tool_name_map.get("computer")
        if not computer_tool_name or not any(c.name == computer_tool_name for c in tool_calls):
            return list(await self._format_function_results(tool_calls, tool_results))

        remaining_calls: list[MCPToolCall] = []
        remaining_results: list[MCPToolResult] = []
        computer_outputs: list[ComputerCallOutput] = []
        ordering: list[tuple[str, int]] = []

        for call, result in zip(tool_calls, tool_results, strict=False):
            if call.name == computer_tool_name:
                screenshot = self._extract_latest_screenshot(result)
                if not screenshot:
                    raise ValueError(
                        "Computer tool result missing screenshot. "
                        "The tool must always return a screenshot for computer_call_output."
                    )
                call_id = call.id or self.pending_call_id
                if not call_id:
                    self.console.warning_log("Computer tool call missing ID; skipping output.")
                    continue
                acknowledged_checks: list[ComputerCallOutputAcknowledgedSafetyCheck] = []
                for check in self.pending_safety_checks:
                    if hasattr(check, "model_dump"):
                        acknowledged_checks.append(check.model_dump())  # type: ignore[arg-type]
                    elif isinstance(check, dict):
                        acknowledged_checks.append(check)  # type: ignore[arg-type]
                output_payload = ComputerCallOutput(
                    type="computer_call_output",
                    call_id=call_id,
                    output=ResponseComputerToolCallOutputScreenshotParam(
                        type="computer_screenshot",
                        image_url=f"data:image/png;base64,{screenshot}",
                    ),
                )
                if acknowledged_checks:
                    output_payload["acknowledged_safety_checks"] = acknowledged_checks
                computer_outputs.append(output_payload)
                self.pending_call_id = None
                self.pending_safety_checks = []
                ordering.append(("computer", len(computer_outputs) - 1))
            else:
                remaining_calls.append(call)
                remaining_results.append(result)
                ordering.append(("function", len(remaining_calls) - 1))

        formatted: list[ComputerCallOutput | FunctionCallOutput] = []
        function_outputs: list[FunctionCallOutput] = []
        if remaining_calls:
            function_outputs = await self._format_function_results(
                remaining_calls, remaining_results
            )

        for kind, idx in ordering:
            if kind == "computer" and idx < len(computer_outputs):
                formatted.append(computer_outputs[idx])
            elif kind == "function" and idx < len(function_outputs):
                formatted.append(function_outputs[idx])
        return formatted

    def _extract_latest_screenshot(self, result: MCPToolResult) -> str | None:
        """Extract the latest screenshot from a tool result."""
        if not result.content:
            return None
        for content in reversed(result.content):
            if isinstance(content, types.ImageContent):
                return content.data
            if isinstance(content, types.TextContent) and result.isError:
                self.console.error_log(f"Computer tool error: {content.text}")
        return None

    async def _format_function_results(
        self, tool_calls: list[MCPToolCall], tool_results: list[MCPToolResult]
    ) -> list[FunctionCallOutput]:
        """Convert MCP tool outputs into function call output items."""
        formatted: list[FunctionCallOutput] = []
        for call, result in zip(tool_calls, tool_results, strict=False):
            if not call.id:
                self.console.warning_log(f"Tool '{call.name}' missing call_id; skipping output.")
                continue

            output_items: ResponseFunctionCallOutputItemListParam = []
            if result.isError:
                output_items.append(
                    ResponseInputTextParam(type="input_text", text="[tool_error] true")
                )

            if result.structuredContent is not None:
                output_items.append(
                    ResponseInputTextParam(
                        type="input_text", text=json.dumps(result.structuredContent, default=str)
                    )
                )

            for block in result.content:
                match block:
                    case types.TextContent():
                        output_items.append(
                            ResponseInputTextContentParam(type="input_text", text=block.text)
                        )
                    case types.ImageContent():
                        mime_type = getattr(block, "mimeType", "image/png")
                        output_items.append(
                            ResponseInputImageContentParam(
                                type="input_image",
                                image_url=f"data:{mime_type};base64,{block.data}",
                            )
                        )
                    case types.ResourceLink():
                        output_items.append(
                            ResponseInputFileContentParam(
                                type="input_file", file_url=str(block.uri)
                            )
                        )
                    case types.EmbeddedResource():
                        match block.resource:
                            case types.TextResourceContents():
                                output_items.append(
                                    ResponseInputTextContentParam(
                                        type="input_text", text=block.resource.text
                                    )
                                )
                            case types.BlobResourceContents():
                                output_items.append(
                                    ResponseInputFileContentParam(
                                        type="input_file", file_data=block.resource.blob
                                    )
                                )
                            case _:
                                self.console.warning_log(
                                    f"Unknown resource type: {type(block.resource)}"
                                )
                    case _:
                        self.console.warning_log(f"Unknown content block type: {type(block)}")

            if not output_items:
                output_items.append(ResponseInputTextParam(type="input_text", text=""))

            formatted.append(
                FunctionCallOutput(
                    type="function_call_output", call_id=call.id, output=output_items
                ),
            )
        return formatted
