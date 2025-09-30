"""
HUD Agent Model Provider for Inspect AI

This custom ModelAPI routes all inspect_ai model calls back through the
MCP interface to your HUD agent running on the host machine.

Architecture:
  inspect_ai (Docker) → HUDAgentModel.generate() → /model/generate HTTP endpoint
  → MCP controller → Host agent → Model API → Response back through chain
"""

from typing import Any
import httpx
import logging

from inspect_ai.model import ModelAPI, GenerateConfig, ModelOutput, ChatMessage
from inspect_ai.tool import ToolInfo, ToolChoice
from inspect_ai.model._registry import modelapi

logger = logging.getLogger(__name__)


@modelapi(name="hud")
class HUDAgentModel(ModelAPI):
    """
    Model API that routes generate() calls to a HUD agent via HTTP.

    Usage:
        model="hud/agent"  # Routes to your agent through MCP

    All model generate() calls from inspect_ai will be sent to the
    environment server's /model/generate endpoint, which can then
    route to your external agent.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        agent_url: str = "http://localhost:8000",  # Environment server URL
        **model_args: dict[str, Any],
    ) -> None:
        super().__init__(model_name, base_url, api_key, [], config)
        self.agent_url = agent_url
        self.model_args = model_args
        self.http_client = httpx.AsyncClient(timeout=300.0)

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        """
        Route generate() call through the environment server to external agent.
        """
        # Convert input messages to serializable format
        messages = []
        for msg in input:
            msg_dict = {
                "role": msg.role,
                "content": str(msg.content) if hasattr(msg, 'content') else ""
            }
            messages.append(msg_dict)

        # Prepare the request
        request_data = {
            "messages": messages,
            "tools": [tool.model_dump() if hasattr(tool, 'model_dump') else tool for tool in tools],
            "tool_choice": tool_choice,
            "config": config.model_dump() if hasattr(config, 'model_dump') else {}
        }

        logger.info(f"Routing generate() call to {self.agent_url}/model/generate")
        logger.debug(f"Request: {len(messages)} messages, {len(tools)} tools")

        try:
            # Call the environment server which will route to the agent
            response = await self.http_client.post(
                f"{self.agent_url}/model/generate",
                json=request_data
            )
            response.raise_for_status()

            data = response.json()
            content = data.get("content", "")

            logger.info(f"Received response: {len(content)} characters")

            # Convert response to ModelOutput
            return ModelOutput.from_content(
                model=self.model_name,
                content=content
            )

        except Exception as e:
            logger.error(f"Error calling agent: {e}")
            # Return error as content
            return ModelOutput.from_content(
                model=self.model_name,
                content=f"Error calling agent: {str(e)}"
            )

    async def __aenter__(self):
        await self.http_client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.http_client.__aexit__(exc_type, exc_val, exc_tb)