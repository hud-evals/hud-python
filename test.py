"""Simple integration test with Claude and Environment via HUD Gateway."""

import asyncio
import logging
import random

import dotenv
from anthropic import AsyncAnthropic

import hud
from hud import Environment
from hud.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
dotenv.load_dotenv()

# Create environment connected to HUD browser
env = Environment().connect_hub("test-browser-26")


async def main():
    # Use HUD inference gateway instead of direct Anthropic API
    client = AsyncAnthropic(
        base_url=settings.hud_gateway_url,
        api_key=settings.api_key,
    )

    async with hud.eval(
        env(), group=3, variants={"model": ["claude-sonnet-4-5", "claude-opus-4-5"]}
    ) as ctx:
        # Make a completion call
        response = await client.messages.create(
            model=ctx.variants["model"],
            max_tokens=1024,
            messages=[{"role": "user", "content": "Navigate to https://example.com"}],
            tools=ctx.as_anthropic_tools(),
        )

        print(f"\nResponse: {response.content}")

        # Execute tool calls
        for block in response.content:
            if block.type == "tool_use":
                print(f"\nExecuting tool: {block.name}")
                print(f"Arguments: {block.input}")

                # Can pass the tool_use block directly (auto-detected)
                result = await ctx.call_tool(block)
                print(f"Result: {result}")

        # Set reward
        ctx.reward = float(random.randint(0, 100)) / 100.0
        print(f"\nTrace ID: {ctx.trace_id}")


if __name__ == "__main__":
    asyncio.run(main())
