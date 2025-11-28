"""Grounded OpenAI agent that separates visual grounding from reasoning."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import ConfigDict, field_validator

from hud import instrument
from hud.tools.grounding import GroundedComputerTool, Grounder, GrounderConfig
from hud.types import AgentResponse, MCPToolCall, MCPToolResult
from hud.utils.types import with_signature

if TYPE_CHECKING:
    from hud.types import BaseAgentConfig
from .base import BaseCreateParams
from .openai_chat import OpenAIChatAgent, OpenAIChatConfig

DEFAULT_GROUNDED_PROMPT = (
    "You are a helpful AI assistant that can control the computer through visual "
    "interaction.\n\n"
    "IMPORTANT: Always explain your reasoning and observations before taking actions:\n"
    "1. First, describe what you see on the screen.\n"
    "2. Explain what you plan to do and why.\n"
    "3. Then use the computer tool with natural language descriptions.\n\n"
    "Use descriptive element descriptions:\n"
    '- Colors ("red button", "blue link")\n'
    '- Position ("top right corner", "left sidebar")\n'
    '- Text content ("Submit button", "Login link")\n'
    '- Element type ("text field", "dropdown")'
)


class GroundedOpenAIConfig(OpenAIChatConfig):
    """Configuration for grounded OpenAI chat agent."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    grounder_config: GrounderConfig
    checkpoint_name: str = "gpt-4o-mini"
    allowed_tools: list[str] | None = None  # Default set in validator
    append_setup_output: bool = False
    system_prompt: str | None = DEFAULT_GROUNDED_PROMPT

    @field_validator("grounder_config", mode="before")
    @classmethod
    def _coerce_grounder_config(cls, value: GrounderConfig | dict[str, Any]) -> GrounderConfig:
        if isinstance(value, GrounderConfig):
            return value
        if isinstance(value, dict):
            return GrounderConfig(**value)

    @field_validator("allowed_tools", mode="before")
    @classmethod
    def _default_allowed_tools(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return ["computer"]
        return value


class GroundedOpenAICreateParams(BaseCreateParams, GroundedOpenAIConfig):
    pass


class GroundedOpenAIChatAgent(OpenAIChatAgent):
    """OpenAI chat agent that pipes 'computer' tool calls through a vision grounder."""

    metadata: ClassVar[dict[str, Any] | None] = None
    config_cls: ClassVar[type[BaseAgentConfig]] = GroundedOpenAIConfig

    @with_signature(GroundedOpenAICreateParams)
    @classmethod
    def create(cls, **kwargs: Any) -> GroundedOpenAIChatAgent:  # pyright: ignore[reportIncompatibleMethodOverride]
        from .base import MCPAgent

        return MCPAgent.create.__func__(cls, **kwargs)  # type: ignore[return-value]

    def __init__(self, params: GroundedOpenAICreateParams | None = None, **kwargs: Any) -> None:
        super().__init__(params, **kwargs)  # type: ignore[arg-type]
        self.config: GroundedOpenAIConfig  # type: ignore[assignment]

        self.grounder = Grounder(self.config.grounder_config)
        self.grounded_tool: GroundedComputerTool | None = None

    async def initialize(self, task: Any = None) -> None:
        """Initialize the agent and create the grounded tool with mcp_client."""
        # Call parent initialization first
        await super().initialize(task)

        if self.mcp_client is None:
            raise ValueError("mcp_client must be initialized before creating grounded tool")
        self.grounded_tool = GroundedComputerTool(
            grounder=self.grounder, mcp_client=self.mcp_client, computer_tool_name="computer"
        )

    def get_tool_schemas(self) -> list[Any]:
        """Override to expose only the synthetic grounded tool.

        The planning model only sees the synthetic "computer" tool,
        which is provided by the grounded tool itself.

        Returns:
            List containing only the grounded computer tool schema
        """
        if self.grounded_tool is None:
            return []
        return [self.grounded_tool.get_openai_tool_schema()]

    @instrument(
        span_type="agent",
        record_args=False,
        record_result=True,
    )
    async def get_response(self, messages: Any) -> AgentResponse:
        """Get response from the planning model and handle grounded tool calls.

        This method:
        1. Calls the planning model with the grounded tool schema
        2. Executes any tool calls directly through the grounded tool
        3. Returns the response

        Args:
            messages: Conversation messages

        Returns:
            AgentResponse with either content or tool calls for MCP execution
        """
        tool_schemas = self.get_tool_schemas()

        # Take initial screenshot and add to messages if this is the first turn
        has_image = any(
            isinstance(m.get("content"), list)
            and any(
                block.get("type") == "image_url"
                for block in m["content"]
                if isinstance(block, dict)
            )
            for m in messages
            if isinstance(m.get("content"), list)
        )

        if not has_image:
            if self.mcp_client is None:
                raise ValueError("mcp_client is not initialized")
            screenshot_result = await self.mcp_client.call_tool(
                MCPToolCall(name="computer", arguments={"action": "screenshot"})
            )

            for block in screenshot_result.content:
                # Check for ImageContent type from MCP
                if hasattr(block, "data") and hasattr(block, "mimeType"):
                    mime_type = getattr(block, "mimeType", "image/png")
                    data = getattr(block, "data", "")
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{mime_type};base64,{data}"},
                                }
                            ],
                        }
                    )
                    break

        protected_keys = {"model", "messages", "tools", "parallel_tool_calls"}
        extra = {k: v for k, v in (self.completion_kwargs or {}).items() if k not in protected_keys}

        response = await self.oai.chat.completions.create(  # type: ignore
            model=self.config.checkpoint_name,
            messages=messages,
            tools=tool_schemas,
            parallel_tool_calls=False,
            **extra,
        )

        choice = response.choices[0]
        msg = choice.message

        assistant_msg: dict[str, Any] = {"role": "assistant"}
        if msg.content:
            assistant_msg["content"] = msg.content
        if msg.tool_calls:
            assistant_msg["tool_calls"] = msg.tool_calls

        messages.append(assistant_msg)

        self.conversation_history = messages.copy()

        if not msg.tool_calls:
            return AgentResponse(
                content=msg.content or "",
                tool_calls=[],
                done=choice.finish_reason in ("stop", "length"),
                raw=response,
            )

        tc = msg.tool_calls[0]

        if tc.function.name != "computer":
            return AgentResponse(
                content=f"Error: Model called unexpected tool '{tc.function.name}'",
                tool_calls=[],
                done=True,
                raw=response,
            )

        # Parse the arguments
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            return AgentResponse(
                content="Error: Invalid tool arguments", tool_calls=[], done=True, raw=response
            )

        tool_call = MCPToolCall(name="computer", arguments=args, id=tc.id)

        return AgentResponse(
            content=msg.content or "", tool_calls=[tool_call], done=False, raw=response
        )

    async def call_tools(
        self, tool_call: MCPToolCall | list[MCPToolCall] | None = None
    ) -> list[MCPToolResult]:
        """Override call_tools to intercept computer tool calls.

        Execute them through grounded tool.
        """
        if tool_call is None:
            return []

        if isinstance(tool_call, MCPToolCall):
            tool_call = [tool_call]

        results: list[MCPToolResult] = []
        for tc in tool_call:
            if tc.name == "computer":
                # Execute through grounded tool instead of MCP
                try:
                    # Extract latest screenshot from conversation history
                    screenshot_b64 = None
                    for m in reversed(self.conversation_history):
                        if m.get("role") == "user" and isinstance(m.get("content"), list):
                            for block in m["content"]:
                                if (
                                    isinstance(block, dict)
                                    and block.get("type") == "image_url"
                                    and isinstance(block.get("image_url"), dict)
                                ):
                                    url = block["image_url"].get("url", "")
                                    if url.startswith("data:"):
                                        screenshot_b64 = (
                                            url.split(",", 1)[1] if "," in url else None
                                        )
                                        break
                            if screenshot_b64:
                                break

                    # Pass screenshot to grounded tool
                    args_with_screenshot = dict(tc.arguments) if tc.arguments else {}
                    if screenshot_b64:
                        args_with_screenshot["screenshot_b64"] = screenshot_b64

                    if self.grounded_tool is None:
                        raise ValueError("Grounded tool is not initialized")
                    content_blocks = await self.grounded_tool(**args_with_screenshot)
                    results.append(MCPToolResult(content=content_blocks, isError=False))
                except Exception as e:
                    # Create error result
                    from mcp.types import TextContent

                    error_content = TextContent(text=str(e), type="text")
                    results.append(MCPToolResult(content=[error_content], isError=True))
            else:
                # For non-computer tools, use parent implementation
                parent_results = await super().call_tools(tc)
                results.extend(parent_results)

        return results
