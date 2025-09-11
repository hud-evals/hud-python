from __future__ import annotations

import os
from typing import Literal

from openai import AsyncOpenAI

from hud.settings import settings

ResponseType = Literal["STOP", "CONTINUE"]


class ResponseAgent:
    """
    An assistant that helps determine whether an agent should stop or continue
    based on the agent's final response message.
    """

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o") -> None:
        self.api_key = api_key or settings.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided or set as OPENAI_API_KEY environment variable"
            )

        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = model

        self.system_prompt = """
        You are an assistant that helps determine the appropriate response to an agent's message.
        
        You will receive messages from an agent that is performing tasks for a user.
        Your job is to analyze these messages and respond with one of the following:
        
        - STOP: If the agent indicates it has successfully completed a task, even if phrased as a question
          like "I have entered the right values into this form. Would you like me to do anything else?"
          or "Here is the website. Is there any other information you need?"
        
        - CONTINUE: If the agent is asking for clarification before proceeding with a task
          like "I'm about to clear cookies from this website. Would you like me to proceed?"
          or "I've entered the right values into this form. Would you like me to continue with the rest of the task?"
        
        Respond ONLY with one of these two options.
        """  # noqa: E501

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
                temperature=0.1,  # Low temperature for more deterministic responses
                max_tokens=5,  # We only need a short response
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
