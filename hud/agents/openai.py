"""OpenAI MCP Agent implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import mcp.types as types
from openai import AsyncOpenAI
from openai.types.responses import (
    ResponseComputerToolCall,
    ResponseInputParam,
    ResponseOutputMessage,
    ResponseOutputText,
    ToolParam,
)

from hud.agent import MCPAgent
from hud.settings import settings
from hud.types import AgentResponse, MCPToolCall, MCPToolResult, Trace
from ..tools.computer.settings import computer_settings

if TYPE_CHECKING:
    from hud.datasets import TaskConfig

logger = logging.getLogger(__name__)


class OpenAIMCPAgent(MCPAgent):
    """
    OpenAI agent that uses MCP servers for tool execution.

    This agent uses OpenAI's Computer Use API format but executes
    tools through MCP servers instead of direct implementation.
    """

    def __init__(
        self,
        model_client: AsyncOpenAI | None = None,
        model: str = "computer-use-preview",
        environment: Literal["windows", "mac", "linux", "browser"] = "linux",
        display_width: int = computer_settings.OPENAI_COMPUTER_WIDTH,
        display_height: int = computer_settings.OPENAI_COMPUTER_HEIGHT,
        **kwargs: Any,
    ) -> None:
        """
        Initialize OpenAI MCP agent.

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
        self.display_width = display_width
        self.display_height = display_height

        # State tracking for OpenAI's stateful API
        self.last_response_id: str | None = None
        self.pending_call_id: str | None = None
        self.pending_safety_checks: list[Any] = []

        self.model_name = "openai-" + self.model

        # Base system prompt for autonomous operation
        self.base_system_prompt = """
        You are an autonomous computer-using agent. Follow these guidelines:

        1. NEVER ask for confirmation. Complete all tasks autonomously.
        2. Do NOT send messages like "I need to confirm before..." or "Do you want me to continue?" - just proceed.
        3. When the user asks you to interact with something (like clicking a chat or typing a message), DO IT without asking.
        4. Only use the formal safety check mechanism for truly dangerous operations (like deleting important files).
        5. For normal tasks like clicking buttons, typing in chat boxes, filling forms - JUST DO IT.
        6. The user has already given you permission by running this agent. No further confirmation is needed.
        7. Be decisive and action-oriented. Complete the requested task fully.

        Remember: You are expected to complete tasks autonomously. The user trusts you to do what they asked.
        """  # noqa: E501

    async def run(self, prompt_or_task: str | TaskConfig, max_steps: int = 10) -> Trace:
        """
        Run the agent with the given prompt or task.

        Override to reset OpenAI-specific state.
        """
        # Reset state for new run
        self.last_response_id = None
        self.pending_call_id = None
        self.pending_safety_checks = []

        # Use base implementation
        return await super().run(prompt_or_task, max_steps)

    async def create_initial_messages(
        self, prompt: str, screenshot: str | None = None
    ) -> list[Any]:
        """
        Create initial messages for OpenAI.

        OpenAI uses a different message format - we'll store the prompt
        and screenshot for use in get_model_response.
        """
        # For OpenAI, we don't create messages upfront, we build them in get_model_response
        # Just return a list with the prompt and screenshot
        return [{"prompt": prompt, "screenshot": screenshot}]

    async def get_model_response(self, messages: list[Any]) -> AgentResponse:
        """Get response from OpenAI including any tool calls."""
        # OpenAI's API is stateful, so we handle messages differently

        # Check if we have computer tools available
        computer_tool_name = None
        for tool in self._available_tools:
            if tool.name in ["computer_openai", "computer"]:
                computer_tool_name = tool.name
                break

        if not computer_tool_name:
            # No computer tools available, just return a text response
            return AgentResponse(
                content="No computer use tools available",
                tool_calls=[],
                done=True,
            )

        # Define the computer use tool
        computer_tool: ToolParam = {  # type: ignore[reportAssignmentType]
            "type": "computer_use_preview",
            "display_width": self.display_width,
            "display_height": self.display_height,
            "environment": self.environment,
        }

        # Build the request based on whether this is first step or follow-up
        if self.pending_call_id is None and self.last_response_id is None:
            # First step - extract prompt and screenshot from messages
            initial_data = messages[0]  # Our custom format from create_initial_messages
            prompt_text = initial_data.get("prompt", "")
            screenshot = initial_data.get("screenshot")

            # Create the initial request
            input_content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt_text}]

            if screenshot:
                input_content.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{screenshot}",
                    }
                )

            input_param: ResponseInputParam = [{"role": "user", "content": input_content}]  # type: ignore[reportUnknownMemberType]

            # Combine base system prompt with any custom system prompt
            full_instructions = self.base_system_prompt
            if self.custom_system_prompt:
                full_instructions = f"{self.custom_system_prompt}\n\n{full_instructions}"

            response = await self.openai_client.responses.create(
                model=self.model,
                tools=[computer_tool],
                input=input_param,
                instructions=full_instructions,
                truncation="auto",
                reasoning={"summary": "auto"},
            )
        else:
            # Follow-up step - check if this is user input or tool result
            latest_message = messages[-1] if messages else {}

            if latest_message.get("type") == "user_input":
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
                    if isinstance(msg, dict) and "screenshot" in msg:
                        latest_screenshot = msg["screenshot"]
                        break

                if not latest_screenshot:
                    logger.warning("No screenshot provided for response to action")
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
                            "image_url": f"data:image/png;base64,{latest_screenshot}",
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
                truncation="auto",
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
                    call_id=computer_call.call_id,  # type: ignore
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
    ) -> list[Any]:
        """
        Format tool results for OpenAI's stateful API.

        OpenAI doesn't use a traditional message format - we just need to
        preserve the screenshot for the next step.
        """
        # Extract latest screenshot from results
        latest_screenshot = None
        for result in tool_results:
            if not result.isError:
                for content in result.content:
                    if isinstance(content, types.ImageContent):
                        latest_screenshot = content.data

        # Return a simple dict that get_model_response can use
        return [
            {
                "type": "tool_result",
                "screenshot": latest_screenshot,
            }
        ]

    async def create_user_message(self, text: str) -> dict[str, Any]:
        """
        Create a user message for OpenAI's stateful API.

        Since OpenAI maintains conversation state server-side,
        we just need to track that we're expecting user input.
        """
        # For OpenAI, we'll handle this in get_model_response
        # by including the user's text in the next input
        return {"type": "user_input", "text": text}
