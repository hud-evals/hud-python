from __future__ import annotations

from typing import Literal

from openai import AsyncOpenAI

from hud.settings import settings

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


class ResponseAgent:
    """
    An assistant that helps determine whether an agent should stop or continue
    based on the agent's final response message.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        system_prompt: str | None = None,
    ) -> None:
        """
        Initialize the ResponseAgent.

        Args:
            model: The model to use via HUD inference gateway (default: "gpt-4o").
                   Supports any model available through inference.hud.ai.
            system_prompt: Optional custom system prompt for determining responses.
        """
        api_key = settings.api_key
        if not api_key:
            raise ValueError(
                "HUD API key is required for auto_respond. Set HUD_API_KEY environment variable."
            )

        self.client: AsyncOpenAI = AsyncOpenAI(
            base_url=settings.hud_gateway_url,
            api_key=api_key,
        )
        self.model = model
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    async def determine_response(self, agent_message: str) -> ResponseType:
        """
        Determine whether the agent should stop or continue based on its message.

        Args:
            agent_message: The message from the agent

        Returns:
            ResponseType: Either "STOP" or "CONTINUE"
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": f"Agent message: {agent_message}\n\nWhat is the appropriate response?",  # noqa: E501
                    },
                ],
                temperature=0.1,
                max_tokens=5,
            )

            response_text = response.choices[0].message.content
            if not response_text:
                return "CONTINUE"

            response_text = response_text.strip().upper()

            # Validate the response
            if "STOP" in response_text:
                return "STOP"
            else:
                return "CONTINUE"

        except Exception:
            return "CONTINUE"  # Default to continue on error
