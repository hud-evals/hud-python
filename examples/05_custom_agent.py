"""
Example: Custom agent using HUD Gateway for inference.

This demonstrates building a custom MCPAgent that:
1. Uses the HUD Gateway (https://inference.hud.ai) for inference
2. Has instrumented get_response() for tracing
3. Works with any model available via the gateway

Usage:
    HUD_API_KEY=sk-hud-... python examples/custom_gateway_agent.py
"""

import asyncio
import json
import os
from typing import Any

import mcp.types as types
from openai import AsyncOpenAI

from hud import instrument
from hud.agents.base import MCPAgent
from hud.datasets import Task
from hud.settings import settings
from hud.types import AgentResponse, MCPToolCall, MCPToolResult


class MyAgent(MCPAgent):
    """
    Custom agent that uses HUD Gateway for inference.

    The HUD Gateway (https://inference.hud.ai) provides:
    - Unified access to Anthropic, OpenAI, Gemini, OpenRouter models
    - Automatic billing via HUD credits
    - No need for individual provider API keys

    All inference calls are traced via @instrument decorator.
    """

    def __init__(
        self,
        checkpoint_name: str = "anthropic/claude-sonnet-4-5-20250929",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.checkpoint_name = checkpoint_name
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Validate API key
        if not settings.api_key:
            raise ValueError("HUD_API_KEY is required for HUD Gateway access")

        # Create OpenAI-compatible client pointing to HUD Gateway
        self.client = AsyncOpenAI(
            base_url=settings.hud_gateway_url,  # https://inference.hud.ai
            api_key=settings.api_key,
        )

    async def get_system_messages(self) -> list[dict[str, Any]]:
        """Return system prompt formatted for OpenAI chat API."""
        system_text = self.system_prompt or "You are a helpful assistant."
        return [{"role": "system", "content": system_text}]

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Convert MCP tools to OpenAI function format."""
        tools = self.get_available_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            }
            for tool in tools
        ]

    @instrument(
        span_type="agent",
        record_args=False,
        record_result=True,
    )
    async def get_response(self, messages: list[Any]) -> AgentResponse:
        """
        Get response from model via HUD Gateway.

        This method is instrumented with @hud.instrument to automatically:
        - Create a span for this inference call
        - Record the response for tracing
        - Track token usage and latency
        """
        tools = self.get_tool_schemas()

        try:
            response = await self.client.chat.completions.create(
                model=self.checkpoint_name,
                messages=messages,
                tools=tools if tools else None,  # type: ignore
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception as e:
            self.console.error_log(f"Gateway inference error: {e}")
            return AgentResponse(
                content=f"Error: {e}",
                tool_calls=[],
                done=True,
                isError=True,
                raw=None,
            )

        choice = response.choices[0]
        msg = choice.message

        # Log usage info
        if response.usage:
            self.console.info_log(
                f"Tokens: {response.usage.prompt_tokens} prompt, "
                f"{response.usage.completion_tokens} completion"
            )

        # Build assistant message for history
        assistant_msg: dict[str, Any] = {"role": "assistant"}
        if msg.content:
            assistant_msg["content"] = msg.content
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},  # type: ignore[union-attr]
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        # Parse tool calls
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)  # type: ignore[union-attr]
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(
                    MCPToolCall(id=tc.id, name=tc.function.name, arguments=args)  # type: ignore[union-attr]
                )

        return AgentResponse(
            content=msg.content or "",
            tool_calls=tool_calls,
            done=choice.finish_reason == "stop" and not tool_calls,
            isError=False,
            raw=response,
        )

    async def format_blocks(self, blocks: list[types.ContentBlock]) -> list[Any]:
        """Format content blocks into OpenAI message format."""
        content_parts = []
        for block in blocks:
            if isinstance(block, types.TextContent):
                content_parts.append({"type": "text", "text": block.text})
            elif isinstance(block, types.ImageContent):
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{block.mimeType};base64,{block.data}"},
                    }
                )
        return [{"role": "user", "content": content_parts}]

    async def format_tool_results(
        self, tool_calls: list[MCPToolCall], tool_results: list[MCPToolResult]
    ) -> list[Any]:
        """Format tool results for the model."""
        messages = []
        for tc, result in zip(tool_calls, tool_results):
            content = ""
            if result.content:
                for block in result.content:
                    if isinstance(block, types.TextContent):
                        content += block.text
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": content or "Tool executed successfully",
                }
            )
        return messages


async def main():
    """Example usage of MyAgent."""

    # Create agent with Claude via Gateway
    agent = MyAgent(
        checkpoint_name="anthropic/claude-sonnet-4-5-20250929",
        max_tokens=2048,
        temperature=0.5,
        verbose=True,
    )

    # Define a task with HUD MCP environment
    task = Task(
        prompt="Go to example.com and tell me the page title",
        mcp_config={
            "hud": {
                "url": "https://mcp.hud.ai/v3/mcp",
                "headers": {
                    "Authorization": f"Bearer {os.environ.get('HUD_API_KEY', '')}",
                    "Mcp-Image": "hudpython/hud-remote-browser:latest",
                },
            }
        },
    )

    # Run the agent - traces are automatically captured
    print("Running agent with HUD Gateway inference...")
    result = await agent.run(task, max_steps=5)

    print("\n=== Results ===")
    print(f"Done: {result.done}")
    print(f"Reward: {result.reward}")
    print(f"Steps: {len(result)}")

    # View traces at https://hud.ai/home


if __name__ == "__main__":
    asyncio.run(main())
