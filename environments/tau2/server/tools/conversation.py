"""Conversation tool for multi-turn user interaction in tau2-bench.

This tool wraps TAU2-bench's UserSimulator to enable multi-turn conversations
within HUD's tool-calling framework.
"""

import os
import logging
from typing import Optional
from hud.tools.base import BaseTool
from mcp.types import TextContent
from server.setup.load import get_tau2_task
from tau2.data_model.message import AssistantMessage, MultiToolMessage
from tau2.user.user_simulator import UserSimulator
from tau2.user.base import STOP, TRANSFER, OUT_OF_SCOPE
from task._system_prompt import _format_system_prompt

logger = logging.getLogger(__name__)


class ConversationTool(BaseTool):
    """
    Send a message to the simulated user and receive their response.

    This tool enables multi-turn conversations in tau2-bench by wrapping
    TAU2's UserSimulator. The agent can send messages to a simulated customer
    (powered by GPT-4) who responds based on the task scenario.

    The conversation continues until the user responds with:
    - "###STOP###" - Task complete, user is satisfied
    - "###TRANSFER###" - User wants to speak to a human
    - "###OUT-OF-SCOPE###" - Scenario doesn't provide enough information
    """

    # Class-level state shared across all instances
    _user_simulator: Optional[UserSimulator] = None
    _user_state = None
    _conversation_history: list[tuple[str, str]] = []  # List of (assistant_msg, user_msg) tuples

    def __init__(self):
        super().__init__(
            env=None,
            name="send_message",
            description=(
                "Send a message to the customer and receive their response. "
                "Use this to communicate with the customer during the conversation. "
                "The customer will respond naturally based on their needs. "
                "When the customer is satisfied, they will indicate the conversation should end."
            ),
        )

    async def __call__(self, message: str) -> list[TextContent]:
        """Send a message to the user simulator and get their response."""
        tau2_task = get_tau2_task()

        # Validate initialization
        if not tau2_task.is_initialized():
            return [
                TextContent(
                    type="text", text="Error: Environment not initialized. Call setup/load first."
                )
            ]

        try:
            # Lazy initialization fallback (in case class variable was reset)
            if self.__class__._user_simulator is None:
                self.__class__.initialize_global(tau2_task)
            # Create and log agent message
            agent_message = AssistantMessage(role="assistant", content=message, cost=0.0)
            tau2_task.add_message(agent_message)

            # Generate user response
            user_message, new_state = self.__class__._user_simulator.generate_next_message(
                message=agent_message, state=self.__class__._user_state
            )
            self.__class__._user_state = new_state
            tau2_task.add_message(user_message)

            # Handle user tool calls (following TAU2 orchestrator pattern)
            # If user makes tool calls, execute them and get next user response
            while user_message.is_tool_call():
                # Execute each tool call via environment
                tool_messages = []
                for tool_call in user_message.tool_calls:
                    tool_msg = tau2_task.environment.get_response(tool_call)
                    tool_messages.append(tool_msg)
                    tau2_task.add_message(tool_msg)

                # Package tool results and send back to user simulator
                if len(tool_messages) > 1:
                    multi_tool_msg = MultiToolMessage(role="tool", tool_messages=tool_messages)
                    user_message, new_state = self.__class__._user_simulator.generate_next_message(
                        message=multi_tool_msg, state=self.__class__._user_state
                    )
                else:
                    user_message, new_state = self.__class__._user_simulator.generate_next_message(
                        message=tool_messages[0], state=self.__class__._user_state
                    )

                self.__class__._user_state = new_state
                tau2_task.add_message(user_message)

            # Format response based on conversation signals
            user_content = user_message.content or ""

            # Record this exchange in conversation history
            self.__class__._conversation_history.append((message, user_content))

            # Get policy and format system prompt (mimics TAU2's system message + conversation pattern)
            system_prompt = ""
            try:
                if tau2_task.environment and hasattr(tau2_task.environment, 'get_policy'):
                    policy = tau2_task.environment.get_policy()
                    # Use TAU2's exact system prompt formatting
                    system_prompt = _format_system_prompt(policy, solo_mode=False)
                    logger.info(f"System prompt generated: {len(system_prompt)} characters")
                else:
                    logger.warning("Environment has no get_policy method")
            except Exception as e:
                logger.error(f"Error generating system prompt: {e}")

            # Build full conversation history string with system prompt (like TAU2's system + messages)
            history_lines = []

            # Add system prompt at the top (mimics TAU2's system message)
            if system_prompt:
                history_lines.append(system_prompt)
                history_lines.append("")
                history_lines.append("=== Conversation History ===")
            else:
                history_lines.append("=== Conversation History ===")

            # Add conversation turns
            for i, (asst_msg, usr_msg) in enumerate(self.__class__._conversation_history, 1):
                history_lines.append(f"[Turn {i}]")
                history_lines.append(f"Assistant: {asst_msg}")
                history_lines.append(f"User: {usr_msg}")
                history_lines.append("")  # Blank line between turns

            conversation_history = "\n".join(history_lines).strip()

            # Format current exchange
            current_exchange = f"Assistant: {message}\n\nUser: {user_content}"

            # Build full response with policy + history (mimics TAU2's architecture)
            full_response = f"{conversation_history}\n\n=== Current Exchange ===\n{current_exchange}"

            if STOP in user_content:
                return [
                    TextContent(
                        type="text",
                        text=f"{full_response}\n\n[User has ended the conversation. Call evaluate/evaluate_task to evaluate your performance.]",
                    )
                ]

            if TRANSFER in user_content:
                return [
                    TextContent(
                        type="text",
                        text=f"{full_response}\n\n[User requested transfer to human agent.]",
                    )
                ]

            if OUT_OF_SCOPE in user_content:
                return [
                    TextContent(
                        type="text",
                        text=f"{full_response}\n\n[User scenario doesn't provide enough information to continue.]",
                    )
                ]

            return [TextContent(type="text", text=full_response)]

        except Exception as e:
            import traceback

            return [
                TextContent(
                    type="text", text=f"Error in conversation: {str(e)}\n\n{traceback.format_exc()}"
                )
            ]

    @classmethod
    def initialize_global(cls, tau2_task):
        """Initialize the class-level user simulator with task scenario."""
        task = tau2_task.task
        if not task or not task.user_scenario:
            raise ValueError("Task must have user_scenario for multi-turn conversation mode.")

        # Reset conversation history for new task
        cls._conversation_history = []

        # Get user LLM configuration
        user_llm = os.getenv("USER_LLM", "gpt-4-0613")
        user_llm_args = {
            "temperature": float(os.getenv("USER_TEMPERATURE", "0.7")),
            "max_tokens": int(os.getenv("USER_MAX_TOKENS", "2500")),
        }

        # Get user tools if available (for telecom domain)
        user_tools = None
        if tau2_task.environment and hasattr(tau2_task.environment, "get_user_tools"):
            try:
                user_tools = tau2_task.environment.get_user_tools()
            except (ValueError, AttributeError):
                # No user tools available for this domain
                user_tools = None

        # Create UserSimulator
        cls._user_simulator = UserSimulator(
            tools=user_tools,
            instructions=task.user_scenario.instructions,
            llm=user_llm,
            llm_args=user_llm_args,
        )
        cls._user_state = cls._user_simulator.get_init_state(message_history=[])


def create_conversation_tool() -> ConversationTool:
    """Factory function to create a ConversationTool instance."""
    return ConversationTool()
