from __future__ import annotations

import logging
from functools import cache
from typing import Literal

import mcp.types as types
from openai import AsyncOpenAI
from openai.types.responses import ResponseOutputText

from hud.settings import settings
from hud.telemetry import instrument

logger = logging.getLogger(__name__)

ResponseType = Literal["STOP", "CONTINUE"]

#: Affirmative reply injected when the classifier decides to continue. The agent
#: paused asking for confirmation; this stands in for the user saying "yes".
CONTINUE_MESSAGE = "Yes, please continue."

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

    # The continue/stop classifier runs through the HUD gateway, so a HUD key is
    # required. Fail loud instead of silently no-opping under BYOK-only configs.
    if not settings.api_key:
        raise ValueError(
            "auto_respond requires HUD_API_KEY: its continue/stop classifier runs "
            "through the HUD gateway.",
        )

    try:
        decision = await _determine_response(content)
    except Exception as exc:
        logger.warning("Auto-respond classifier call failed: %s", exc)
        return None

    if decision == "STOP":
        return None

    return types.PromptMessage(
        role="user",
        content=types.TextContent(text=CONTINUE_MESSAGE, type="text"),
    )


@cache
def _client() -> AsyncOpenAI:
    api_key = settings.api_key
    if not api_key:
        raise ValueError(
            "HUD API key is required for auto_respond. Set HUD_API_KEY environment variable."
        )

    return AsyncOpenAI(
        base_url=settings.hud_gateway_url,
        api_key=api_key,
    )


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

    # The classifier is instructed to reply with exactly STOP or CONTINUE. Treat
    # only an exact CONTINUE as continue; empty/ambiguous output is fail-safe STOP.
    response_text = "".join(text_parts).strip().upper()
    return "CONTINUE" if response_text == "CONTINUE" else "STOP"
