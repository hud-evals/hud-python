"""ByteDance Ark agent for HUD evaluations.

This agent connects to ByteDance's Ark API (OpenAI-compatible) and supports
multiple "think modes" for controlling the model's reasoning verbosity.

Usage:
    from hud.agents import ByteDanceAgent

    agent = ByteDanceAgent.create(
        checkpoint_name="ep-20251106141738-cn2tk",
        think_mode="efficient",
    )
    result = await agent.run(task, max_steps=10)

Environment variables:
    ARK_API_KEY: ByteDance Ark API key (required)
    BYTEDANCE_THINK_MODE: Default think mode (optional, defaults to "efficient")
    BYTEDANCE_MAX_TOKENS: Max tokens per response (optional, defaults to 16384)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, ClassVar

from openai import AsyncOpenAI
from pydantic import ConfigDict, Field

from hud import instrument
from hud.settings import settings
from hud.types import AgentResponse, BaseAgentConfig, MCPToolCall
from hud.utils.types import with_signature

from .openai_chat import OpenAIChatAgent, OpenAIChatConfig, OpenAIChatCreateParams

logger = logging.getLogger(__name__)

# ByteDance Ark API configuration
ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

# Curated allowlist of ByteDance model endpoints
BYTEDANCE_ALLOWED_MODELS: list[str] = [
    "ep-20251106141738-cn2tk",  # Agentic Model (GUI, Browser, MCP/API)
]

# Think mode system prompts - these control how the model reasons
THINK_MODE_PROMPTS: dict[str, str] = {
    "no_think": """You should begin by detailing the internal reasoning process, and then
present the answer to the user. The reasoning process should be enclosed
within <think_never_used_51bce0c785ca2f68081bfa7d91973934> and
</think_never_used_51bce0c785ca2f68081bfa7d91973934> tags, as follows:
<think_never_used_51bce0c785ca2f68081bfa7d91973934> reasoning process here
</think_never_used_51bce0c785ca2f68081bfa7d91973934> answer here.

You have different modes of thinking:

Unrestricted think mode: Engage in an internal thinking process with
thorough reasoning and reflections. You have an unlimited budget for
thinking tokens and can continue thinking until you fully solve the
problem.

Efficient think mode: Provide a concise internal thinking process with
efficient reasoning and reflections. You don't have a strict token budget
but be less verbose and more direct in your thinking.

No think mode: Respond directly to the question without any internal
reasoning process or extra thinking tokens. Still follow the template with
the minimum required thinking tokens to justify the answer.

Budgeted think mode: Limit your internal reasoning and reflections to stay
within the specified token budget

Based on the complexity of the problem, select the appropriate mode for
reasoning among the provided options listed below.

Provided Mode(s):
No think""",

    "efficient": """You should begin by detailing the internal reasoning process, and then
present the answer to the user. The reasoning process should be enclosed
within <think_never_used_51bce0c785ca2f68081bfa7d91973934> and
</think_never_used_51bce0c785ca2f68081bfa7d91973934> tags, as follows:
<think_never_used_51bce0c785ca2f68081bfa7d91973934> reasoning process here
</think_never_used_51bce0c785ca2f68081bfa7d91973934> answer here.

You have different modes of thinking:

Unrestricted think mode: Engage in an internal thinking process with
thorough reasoning and reflections. You have an unlimited budget for
thinking tokens and can continue thinking until you fully solve the
problem.

Efficient think mode: Provide a concise internal thinking process with
efficient reasoning and reflections. You don't have a strict token budget
but be less verbose and more direct in your thinking.

No think mode: Respond directly to the question without any internal
reasoning process or extra thinking tokens. Still follow the template with
the minimum required thinking tokens to justify the answer.

Budgeted think mode: Limit your internal reasoning and reflections to stay
within the specified token budget

Based on the complexity of the problem, select the appropriate mode for
reasoning among the provided options listed below.

Provided Mode(s):
Efficient think""",

    "budgeted": """You should begin by detailing the internal reasoning process, and then
present the answer to the user. The reasoning process should be enclosed
within <think_never_used_51bce0c785ca2f68081bfa7d91973934> and
</think_never_used_51bce0c785ca2f68081bfa7d91973934> tags, as follows:
<think_never_used_51bce0c785ca2f68081bfa7d91973934> reasoning process here
</think_never_used_51bce0c785ca2f68081bfa7d91973934> answer here.

You have different modes of thinking:

Unrestricted think mode: Engage in an internal thinking process with
thorough reasoning and reflections. You have an unlimited budget for
thinking tokens and can continue thinking until you fully solve the
problem.

Efficient think mode: Provide a concise internal thinking process with
efficient reasoning and reflections. You don't have a strict token budget
but be less verbose and more direct in your thinking.

No think mode: Respond directly to the question without any internal
reasoning process or extra thinking tokens. Still follow the template with
the minimum required thinking tokens to justify the answer.

Budgeted think mode: Limit your internal reasoning and reflections to stay
within the specified token budget

Based on the complexity of the problem, select the appropriate mode for
reasoning among the provided options listed below.

Provided Mode(s):
Budgeted think""",

    "unrestricted": """You should begin by detailing the internal reasoning process, and then
present the answer to the user. The reasoning process should be enclosed
within <think_never_used_51bce0c785ca2f68081bfa7d91973934> and
</think_never_used_51bce0c785ca2f68081bfa7d91973934> tags, as follows:
<think_never_used_51bce0c785ca2f68081bfa7d91973934> reasoning process here
</think_never_used_51bce0c785ca2f68081bfa7d91973934> answer here.

You have different modes of thinking:

Unrestricted think mode: Engage in an internal thinking process with
thorough reasoning and reflections. You have an unlimited budget for
thinking tokens and can continue thinking until you fully solve the
problem.

Efficient think mode: Provide a concise internal thinking process with
efficient reasoning and reflections. You don't have a strict token budget
but be less verbose and more direct in your thinking.

No think mode: Respond directly to the question without any internal
reasoning process or extra thinking tokens. Still follow the template with
the minimum required thinking tokens to justify the answer.

Budgeted think mode: Limit your internal reasoning and reflections to stay
within the specified token budget

Based on the complexity of the problem, select the appropriate mode for
reasoning among the provided options listed below.

Provided Mode(s):
Unrestricted think""",
}


class ByteDanceConfig(OpenAIChatConfig):
    """Configuration for ByteDanceAgent."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = "ByteDance"
    checkpoint_name: str = "ep-20251106141738-cn2tk"
    think_mode: str | None = None  # None = use BYTEDANCE_THINK_MODE env var
    max_tokens: int | None = None  # None = use BYTEDANCE_MAX_TOKENS env var


class ByteDanceCreateParams(OpenAIChatCreateParams, ByteDanceConfig):
    """Create params for ByteDanceAgent."""

    pass


class ByteDanceAgent(OpenAIChatAgent):
    """MCP-enabled agent using ByteDance Ark API.

    This agent extends OpenAIChatAgent to work with ByteDance's
    OpenAI-compatible Ark API. It supports multiple "think modes" that
    control how verbose the model's reasoning process is.

    Think modes:
        - no_think: Minimal reasoning, direct answers
        - efficient: Concise reasoning (default)
        - budgeted: Limited token budget for reasoning
        - unrestricted: Full reasoning without limits

    Example:
        ```python
        from hud.agents import ByteDanceAgent
        from hud.datasets import Task

        agent = ByteDanceAgent.create(
            checkpoint_name="ep-20251106141738-cn2tk",
            think_mode="efficient",
        )

        task = Task(
            prompt="Search for information about quantum computing",
            mcp_config={"local": {"command": "docker", "args": [...]}},
        )

        result = await agent.run(task, max_steps=10)
        print(f"Reward: {result.reward}")
        ```
    """

    metadata: ClassVar[dict[str, Any]] = {"provider": "bytedance", "api": "ark"}
    config_cls: ClassVar[type[BaseAgentConfig]] = ByteDanceConfig

    @with_signature(ByteDanceCreateParams)
    @classmethod
    def create(cls, **kwargs: Any) -> ByteDanceAgent:  # pyright: ignore[reportIncompatibleMethodOverride]
        from .base import MCPAgent

        return MCPAgent.create.__func__(cls, **kwargs)  # type: ignore[return-value]

    def __init__(self, params: ByteDanceCreateParams | None = None, **kwargs: Any) -> None:
        """Initialize the ByteDance agent."""
        # Handle legacy kwargs pattern
        if params is None:
            params = ByteDanceCreateParams(**kwargs)

        # Get API key from settings or environment
        api_key = params.api_key or settings.ark_api_key or os.environ.get("ARK_API_KEY")
        if not api_key:
            raise ValueError(
                "ARK_API_KEY not set. Please provide api_key, set the ARK_API_KEY "
                "environment variable, or add it to your .env file."
            )

        # Get think mode from params or environment
        think_mode = params.think_mode or os.environ.get("BYTEDANCE_THINK_MODE", "efficient")
        think_mode = think_mode.strip().lower()
        if think_mode not in THINK_MODE_PROMPTS:
            logger.warning(f"Invalid think_mode '{think_mode}', falling back to 'efficient'")
            think_mode = "efficient"

        self.think_mode = think_mode
        self._mode_prompt = THINK_MODE_PROMPTS[think_mode]

        # Get base URL and max tokens
        base_url = params.base_url or os.environ.get("ARK_BASE_URL", ARK_BASE_URL)
        max_tokens = params.max_tokens or int(os.environ.get("BYTEDANCE_MAX_TOKENS", "32768"))

        # Create the OpenAI-compatible client for ByteDance Ark
        client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )

        # Update params with our computed values
        params.openai_client = client
        params.api_key = api_key
        params.base_url = base_url
        # NOTE: Per ByteDance docs, thinking must be disabled at API level
        # when using system prompt-based think modes. We use extra_body since
        # "thinking" is not a standard OpenAI parameter.
        params.completion_kwargs = {
            "max_tokens": max_tokens,
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        params.model_name = "ByteDance"

        # Log at WARNING level so it's always visible
        logger.warning(
            f"🧠 ByteDanceAgent: checkpoint={params.checkpoint_name}, "
            f"THINK_MODE={think_mode.upper()}, max_tokens={max_tokens}"
        )

        # Initialize parent
        super().__init__(params, **kwargs)

    @staticmethod
    def _oai_to_mcp(tool_call: Any) -> MCPToolCall | None:  # type: ignore[valid-type]
        """Convert an OpenAI ``tool_call`` to :class:`MCPToolCall`.

        Returns None if JSON parsing fails (malformed tool call).
        """
        try:
            args = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError as e:
            logger.warning(
                f"Failed to parse tool call arguments: {e}. "
                f"Raw: {tool_call.function.arguments[:100] if tool_call.function.arguments else 'empty'}"
            )
            return None  # Skip malformed tool calls

        if isinstance(args, list):
            args = args[0]
        if not isinstance(args, dict):
            args = {}
        return MCPToolCall(
            id=tool_call.id,
            name=tool_call.function.name,
            arguments=args,
        )

    async def get_system_messages(self) -> list[Any]:
        """Get system messages with think mode prompt appended.

        Returns:
            List containing the system message with think mode instructions.
        """
        base_messages = await super().get_system_messages()

        # Append think mode prompt to system message
        if self._mode_prompt:
            if base_messages:
                base_messages[0]["content"] += f"\n\n{self._mode_prompt}"
            else:
                # If no base system prompt, create one with just the think mode prompt
                base_messages = [{"role": "system", "content": self._mode_prompt}]

        return base_messages

    @instrument(
        span_type="agent",
        record_args=False,
        record_result=True,
    )
    async def get_response(self, messages: list[Any]) -> AgentResponse:
        """Send chat request to ByteDance and convert the response.

        Overrides parent to handle malformed JSON in tool calls gracefully.
        """
        from typing import cast

        from openai.types.chat import ChatCompletionToolParam

        # Convert MCP tool schemas to OpenAI format
        tools = cast("list[ChatCompletionToolParam]", self.get_tool_schemas())

        protected_keys = {"model", "messages", "tools"}
        extra = {k: v for k, v in (self.completion_kwargs or {}).items() if k not in protected_keys}

        try:
            response = await self._invoke_chat_completion(
                messages=messages,
                tools=tools,  # type: ignore
                extra=extra,
            )
        except Exception as e:
            error_content = f"Error getting response {e}"
            if "Invalid JSON" in str(e):
                error_content = "Invalid JSON, response was truncated"
            self.hud_console.warning_log(error_content)

            return AgentResponse(
                content=error_content,
                tool_calls=[],
                done=True,
                isError=True,
                raw=None,
            )

        choice = response.choices[0]
        msg = choice.message
        assistant_msg: dict[str, Any] = {"role": "assistant"}

        if msg.content:
            assistant_msg["content"] = msg.content

        if msg.tool_calls:
            serialized_tool_calls = []
            for tc in msg.tool_calls:
                serialized_tc = {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                serialized_tool_calls.append(serialized_tc)
            assistant_msg["tool_calls"] = serialized_tool_calls

        messages.append(assistant_msg)

        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.function.name is not None:  # type: ignore
                    # _oai_to_mcp returns MCPToolCall or None if malformed
                    mcp_call = self._oai_to_mcp(tc)
                    if mcp_call is not None:
                        tool_calls.append(mcp_call)

        # Handle truncation: if response was cut off, tell the model to retry
        # with less thinking instead of stopping the task
        if choice.finish_reason == "length":
            self.hud_console.warning_log(
                "⚠️ Response was truncated (hit token limit). Asking model to retry."
            )
            # Return error so model can retry with less verbose thinking
            return AgentResponse(
                content="ERROR: Your response was too long and got truncated. "
                        "Please try again.",
                done=False,  # Keep going - let model retry
                isError=True,
                raw=response,
            )

        return AgentResponse(
            content=msg.content or "",
            tool_calls=tool_calls,
            done=False,  # Never auto-stop, let the agent loop handle termination
            raw=response,
        )

