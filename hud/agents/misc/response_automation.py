from __future__ import annotations

import logging
from functools import cache
from typing import TYPE_CHECKING, Literal, cast

import mcp.types as types
from openai.types.responses import ResponseOutputText

from hud.telemetry import instrument

if TYPE_CHECKING:
    from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

ResponseType = Literal["STOP", "CONTINUE"]

DEFAULT_SYSTEM_PROMPT = """\
You are an assistant that helps determine the appropriate response to an agent's message.

You will receive messages from an agent that is performing tasks for a user.
Your job is to analyze these messages and respond with one of the following:

- STOP: If the agent indicates it has successfully completed a task or is stuck,
  struggling or says it cannot complete the task, even if phrased as a question
  like "I have entered the right values into this form. Would you like me to do
  anything else?" or "Here is the website. Is there any other information you
  need?" or if the agent has strongly determined it wants to stop the task like
  "The task is infeasible. Can I help you with something else?"

- CONTINUE: If the agent is asking for clarification before proceeding with a task
  like "I'm about to clear cookies from this website. Would you like me to proceed?"
  or "I've entered the right values into this form. Would you like me to continue
  with the rest of the task?"

Respond ONLY with one of these two options."""


async def auto_respond(
    content: str | None,
    *,
    enabled: bool,
) -> types.PromptMessage | None:
    if not enabled or not content:
        return None

    try:
        decision = await _determine_response(content)
    except Exception as exc:
        logger.warning("Auto-respond failed: %s", exc)
        return None

    if decision == "STOP":
        return None

    return types.PromptMessage(
        role="user",
        content=types.TextContent(text=decision, type="text"),
    )


@cache
def _client() -> AsyncOpenAI:
    from hud.utils.gateway import build_gateway_client

    return cast("AsyncOpenAI", build_gateway_client("openai"))


@instrument(
    category="agent",
    name="response_automation",
    internal_type="user-message",
)
async def _determine_response(
    agent_message: str,
    *,
    model: str = "gpt-5",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> ResponseType:
    response = await _client().responses.create(
        model=model,
        instructions=system_prompt,
        input=[
            {
                "role": "user",
                "content": f"Agent message: {agent_message}\n\nWhat is the appropriate response?",
            },
        ],
        reasoning={"effort": "low"},
        max_output_tokens=256,
        extra_headers={"Trace-Id": ""},
    )

    text_parts: list[str] = []
    for item in response.output:
        if item.type == "message":
            text_parts.extend(
                content.text for content in item.content if isinstance(content, ResponseOutputText)
            )

    response_text = "".join(text_parts)
    if not response_text:
        return "CONTINUE"

    response_text = response_text.strip().upper()
    return "STOP" if "STOP" in response_text else "CONTINUE"
