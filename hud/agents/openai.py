"""OpenAI MCP Agent implementation."""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Literal

import mcp.types as types
from openai import AsyncOpenAI
from openai.types.responses import (
    ResponseComputerToolCall,
    ResponseInputMessageContentListParam,
    ResponseInputParam,
    ResponseOutputMessage,
    ResponseOutputText,
    ToolParam,
)

import hud
from hud.settings import settings
from hud.tools.computer.settings import computer_settings
from hud.types import AgentResponse, MCPToolCall, MCPToolResult, Trace

from .base import MCPAgent

logger = logging.getLogger(__name__)


class OperatorAgent(MCPAgent):
    """
    Operator agent that uses MCP servers for tool execution.

    This agent uses OpenAI's Computer Use API format but executes
    tools through MCP servers instead of direct implementation.
    """

    metadata: ClassVar[dict[str, Any]] = {
        "display_width": computer_settings.OPENAI_COMPUTER_WIDTH,
        "display_height": computer_settings.OPENAI_COMPUTER_HEIGHT,
    }
    required_tools: ClassVar[list[str]] = ["openai_computer"]

    def __init__(
        self,
        model_client: AsyncOpenAI | None = None,
        model: str = "computer-use-preview",
        environment: Literal["windows", "mac", "linux", "browser"] = "linux",
        **kwargs: Any,
    ) -> None:
        """
        Initialize Operator MCP agent.

        Args:
            client: AsyncOpenAI client (created if not provided)
            model: OpenAI model to use
            environment: Environment type for computer use
            display_width: Display width for computer use
            display_height: Display height for computer use
            **kwargs: Additional arguments passed to MCPAgent
        """
        super().__init__(**kwargs)

        # Initialize client if not provided
        if model_client is None:
            api_key = settings.openai_api_key
            if not api_key:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY.")
            model_client = AsyncOpenAI(api_key=api_key)

        self.openai_client = model_client
        self.model = model
        self.environment = environment

        # State tracking for OpenAI's stateful API
        self.last_response_id: str | None = None
        self.pending_call_id: str | None = None
        self.pending_safety_checks: list[Any] = []

        self.model_name = "openai-" + self.model

        # Append OpenAI-specific instructions to the base system prompt
        openai_instructions = """
        You are an autonomous computer-using agent. Follow these guidelines:

        1. NEVER ask for confirmation. Complete all tasks autonomously.
        2. Do NOT send messages like "I need to confirm before..." or "Do you want me to continue?" - just proceed.
        3. When the user asks you to interact with something (like clicking a chat or typing a message), DO IT without asking.
        4. Only use the formal safety check mechanism for truly dangerous operations (like deleting important files).
        5. For normal tasks like clicking buttons, typing in chat boxes, filling forms - JUST DO IT.
        6. The user has already given you permission by running this agent. No further confirmation is needed.
        7. Be decisive and action-oriented. Complete the requested task fully.

        Remember: You are expected to complete tasks autonomously. The user trusts you to do what they asked.
        """.strip()  # noqa: E501

        # Append OpenAI instructions to any base system prompt
        if self.system_prompt:
            self.system_prompt = f"{self.system_prompt}\n\n{openai_instructions}"
        else:
            self.system_prompt = openai_instructions

    async def _run_context(self, context: list[types.ContentBlock], max_steps: int = 10) -> Trace:
        """
        Run the agent with the given prompt or task.

        Override to reset OpenAI-specific state.
        """
        # Reset state for new run
        self.last_response_id = None
        self.pending_call_id = None
        self.pending_safety_checks = []

        # Use base implementation
        return await super()._run_context(context, max_steps=max_steps)

    async def get_system_messages(self) -> list[Any]:
        """
        Create initial messages for OpenAI.

        OpenAI uses a different message format - we'll store the prompt
        and screenshot for use in get_model_response.
        """
        return []

    async def format_blocks(
        self, blocks: list[types.ContentBlock]
    ) -> ResponseInputMessageContentListParam:
        """
        Format blocks for OpenAI input format.

        Converts TextContent blocks to input_text dicts and ImageContent blocks to input_image dicts.
        """  # noqa: E501
        formatted = []
        for block in blocks:
            if isinstance(block, types.TextContent):
                formatted.append({"type": "input_text", "text": block.text})
            elif isinstance(block, types.ImageContent):
                mime_type = getattr(block, "mimeType", "image/png")
                formatted.append(
                    {"type": "input_image", "image_url": f"data:{mime_type};base64,{block.data}"}
                )
        return formatted

    @hud.instrument(
        span_type="agent",
        record_args=False,  # Messages can be large
        record_result=True,
    )
    async def get_response(self, messages: ResponseInputMessageContentListParam) -> AgentResponse:
        """Get response from OpenAI including any tool calls."""
        # OpenAI's API is stateful, so we handle messages differently

        # Get the computer tool (guaranteed to exist due to required_tools)
        computer_tool_name = "openai_computer"

        # Define the computer use tool
        computer_tool: ToolParam = {  # type: ignore[reportAssignmentType]
            "type": "computer_use_preview",
            "display_width": self.metadata["display_width"],
            "display_height": self.metadata["display_height"],
            "environment": self.environment,
        }

        # Build the request based on whether this is first step or follow-up
        if self.pending_call_id is None and self.last_response_id is None:
            # First step - messages are already formatted dicts from format_blocks
            # format_blocks returns type ResponseInputMessageContentListParam, which is a list of dicts  # noqa: E501
            input_content: ResponseInputMessageContentListParam = []

            input_content.extend(messages)

            # If no content was added, add empty text to avoid empty request
            if not input_content:
                input_content.append({"type": "input_text", "text": ""})

            input_param: ResponseInputParam = [{"role": "user", "content": input_content}]  # type: ignore[reportUnknownMemberType]

            response = await self.openai_client.responses.create(
                model=self.model,
                tools=[computer_tool],
                input=input_param,
                instructions=self.system_prompt,
                truncation="auto",
                reasoning={"summary": "auto"},  # type: ignore[arg-type]
            )
        else:
            # Follow-up step - check if this is user input or tool result
            latest_message = messages[-1] if messages else {}

            if latest_message.get("type") == "input_text":
                # User provided input in conversation mode
                user_text = latest_message.get("text", "")
                input_param_followup: ResponseInputParam = [  # type: ignore[reportAssignmentType]
                    {"role": "user", "content": [{"type": "input_text", "text": user_text}]}
                ]
                # Reset pending_call_id since this is user input, not a tool response
                self.pending_call_id = None
            else:
                # Tool result - need screenshot from processed results
                latest_screenshot = None
                for msg in reversed(messages):
                    if isinstance(msg, dict) and "image_url" in msg:
                        latest_screenshot = msg["image_url"]  # type: ignore
                        break

                if not latest_screenshot:
                    self.console.warning_log("No screenshot provided for response to action")
                    return AgentResponse(
                        content="No screenshot available for next action",
                        tool_calls=[],
                        done=True,
                    )

                # Create response to previous action
                input_param_followup: ResponseInputParam = [  # type: ignore[reportAssignmentType]
                    {  # type: ignore[reportAssignmentType]
                        "call_id": self.pending_call_id,
                        "type": "computer_call_output",
                        "output": {
                            "type": "input_image",
                            "image_url": latest_screenshot,
                        },
                        "acknowledged_safety_checks": self.pending_safety_checks,
                    }
                ]

            self.pending_safety_checks = []

            response = await self.openai_client.responses.create(
                model=self.model,
                previous_response_id=self.last_response_id,
                tools=[computer_tool],
                input=input_param_followup,
                instructions=self.system_prompt,
                truncation="auto",
                reasoning={"summary": "auto"},  # type: ignore[arg-type]
            )

        # Store response ID for next call
        self.last_response_id = response.id

        # Process response
        result = AgentResponse(
            content="",
            tool_calls=[],
            done=False,  # Will be set to True only if no tool calls
        )

        self.pending_call_id = None

        # Check for computer calls
        computer_calls = [
            item
            for item in response.output
            if isinstance(item, ResponseComputerToolCall) and item.type == "computer_call"
        ]

        if computer_calls:
            # Process computer calls
            result.done = False
            for computer_call in computer_calls:
                self.pending_call_id = computer_call.call_id
                self.pending_safety_checks = computer_call.pending_safety_checks

                # Convert OpenAI action to MCP tool call
                action = computer_call.action.model_dump()

                # Create MCPToolCall object with OpenAI metadata as extra fields
                # Pyright will complain but the tool class accepts extra fields
                tool_call = MCPToolCall(
                    name=computer_tool_name,
                    arguments=action,
                    id=computer_call.call_id,  # type: ignore
                    pending_safety_checks=computer_call.pending_safety_checks,  # type: ignore
                )
                result.tool_calls.append(tool_call)
        else:
            # No computer calls, check for text response
            for item in response.output:
                if isinstance(item, ResponseOutputMessage) and item.type == "message":
                    # Extract text from content blocks
                    text_parts = [
                        content.text
                        for content in item.content
                        if isinstance(content, ResponseOutputText)
                    ]
                    if text_parts:
                        result.content = "".join(text_parts)
                        break

        # Extract reasoning if present
        reasoning_text = ""
        for item in response.output:
            if item.type == "reasoning" and hasattr(item, "summary") and item.summary:
                reasoning_text += f"Thinking: {item.summary[0].text}\n"

        if reasoning_text:
            result.content = reasoning_text + result.content if result.content else reasoning_text

        # Set done=True if no tool calls (task complete or waiting for user)
        if not result.tool_calls:
            result.done = True

        return result

    async def format_tool_results(
        self, tool_calls: list[MCPToolCall], tool_results: list[MCPToolResult]
    ) -> ResponseInputMessageContentListParam:
        """
        Format tool results for OpenAI's stateful API.

        Tool result content is a list of ContentBlock objects.
        We need to extract the latest screenshot from the tool results.

        This assumes that you only care about computer tool results for your agent loop.
        If you need to add other content, you can do so by adding a new ContentBlock object to the list.

        Returns formatted dicts with tool result data, preserving screenshots.
        """  # noqa: E501
        formatted_results = []
        latest_screenshot = None

        # Extract all content from tool results
        for result in tool_results:
            if result.isError:
                # If it's an error, the error details are in the content
                for content in result.content:
                    if isinstance(content, types.TextContent):
                        # Don't add error text as input_text, just track it
                        self.console.error_log(f"Tool error: {content.text}")
                    elif isinstance(content, types.ImageContent):
                        # Even error results might have images
                        latest_screenshot = content.data
            else:
                # Extract content from successful results
                for content in result.content:
                    if isinstance(content, types.ImageContent):
                        latest_screenshot = content.data
                        break

        # Return a dict with the latest screenshot for the follow-up step
        if latest_screenshot:
            formatted_results.append(
                {"type": "input_image", "image_url": f"data:image/png;base64,{latest_screenshot}"}
            )

        return formatted_results
