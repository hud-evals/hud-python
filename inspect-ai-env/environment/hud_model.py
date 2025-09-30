"""
HUD Agent Model Provider for Inspect AI

This custom ModelAPI wraps HUD agents (ClaudeAgent, OperatorAgent, GenericOpenAIChatAgent)
to make them compatible with Inspect AI's model interface.

Architecture:
  inspect_ai → HUDAgentModel.generate() → HUD Agent.get_response() → ModelOutput
"""

from typing import Any
import logging

from inspect_ai.model import ModelAPI, GenerateConfig, ModelOutput, ChatMessage
from inspect_ai.tool import ToolInfo, ToolChoice
from inspect_ai.model._registry import modelapi

import mcp.types as types
from .null_mcp_client import NullMCPClient
from .agent_factory import create_agent_for_model

logger = logging.getLogger(__name__)


@modelapi(name="hud")
class HUDAgentModel(ModelAPI):
    """
    Model API that wraps HUD agents for use with Inspect AI.

    Usage:
        model="hud/claude-3-5-sonnet"  # Uses ClaudeAgent
        model="hud/gpt-4o"             # Uses GenericOpenAIChatAgent
        model="hud/computer-use-preview"  # Uses OperatorAgent

    The model name after "hud/" is used to select and configure the appropriate agent.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: dict[str, Any],
    ) -> None:
        super().__init__(model_name, base_url, api_key, [], config)
        self.model_args = model_args

        # Extract actual model name from "hud/model-name" format
        self.actual_model_name = model_name.split("/", 1)[1] if "/" in model_name else model_name

        # Create null MCP client (Inspect AI manages tools, not MCP)
        self.mcp_client = NullMCPClient()

        # Create the appropriate HUD agent
        logger.info(f"Initializing HUD agent for model: {self.actual_model_name}")
        self.agent = create_agent_for_model(
            self.actual_model_name,
            mcp_client=self.mcp_client,
            verbose=model_args.get("verbose", False),
            **model_args,
        )

        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Ensure agent is initialized (done lazily on first use)."""
        if not self._initialized:
            await self.mcp_client.initialize()
            # Initialize agent without a task (simple mode)
            await self.agent.initialize(task=None)
            self._initialized = True

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        """
        Generate a response using the HUD agent.

        Converts Inspect AI messages to HUD agent format, calls the agent,
        and converts the response back to Inspect AI format.
        """
        await self._ensure_initialized()

        logger.info(f"Generate called with {len(input)} messages, {len(tools)} tools")

        try:
            # Convert Inspect AI ChatMessage to MCP ContentBlocks
            content_blocks = []
            for msg in input:
                # Handle different message types
                if hasattr(msg, 'content'):
                    if isinstance(msg.content, str):
                        content_blocks.append(types.TextContent(type="text", text=msg.content))
                    elif isinstance(msg.content, list):
                        # Handle multi-part content (text, images, etc.)
                        for part in msg.content:
                            if isinstance(part, str):
                                content_blocks.append(types.TextContent(type="text", text=part))
                            elif hasattr(part, 'text'):
                                content_blocks.append(types.TextContent(type="text", text=part.text))
                            # TODO: Handle image content if needed

            # Format messages for the specific agent
            system_messages = await self.agent.get_system_messages()
            agent_messages = system_messages + await self.agent.format_message(content_blocks)

            logger.debug(f"Calling agent.get_response() with {len(agent_messages)} messages")

            # Call the agent's get_response method
            response = await self.agent.get_response(agent_messages)

            logger.info(f"Agent response: {len(response.content) if response.content else 0} chars")

            # Convert AgentResponse to ModelOutput
            return ModelOutput.from_content(
                model=self.model_name,
                content=response.content or ""
            )

        except Exception as e:
            logger.error(f"Error in HUD agent generate: {e}", exc_info=True)
            # Return error as content
            return ModelOutput.from_content(
                model=self.model_name,
                content=f"Error in agent: {str(e)}"
            )

    async def __aenter__(self):
        await self._ensure_initialized()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cleanup if needed
        if self._initialized and self.mcp_client:
            await self.mcp_client.shutdown()