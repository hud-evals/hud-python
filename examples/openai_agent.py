#!/usr/bin/env python3
"""
OpenAI Agent Example

This example showcases OpenAI-specific features:
- Function calling interface
- Structured outputs
- Reasoning summaries
- OpenAI-specific parameters

OpenAI models excel at function calling and structured reasoning.
"""

import asyncio
import hud
from hud.mcp import OpenAIMCPAgent
from hud.mcp.client import MCPClient


async def main():
    with hud.trace("OpenAI Agent Demo"):
        # Configure for cloud environment
        mcp_config = {
            "hud": {
                "url": "${HUD_MCP_URL}",
                "headers": {
                    "Authorization": "Bearer ${HUD_API_KEY}",
                    "Mcp-Image": "hudpython/hud-browser:latest",
                    "Run-Id": "${RUN_ID}",
                },
            }
        }

        # Create OpenAI-specific agent
        client = MCPClient(mcp_config=mcp_config)
        agent = OpenAIMCPAgent(
            mcp_client=client,
            model="gpt-4o",  # Latest OpenAI model
            temperature=0.3,
            allowed_tools=["openai_computer"],  # OpenAI-specific computer tool
            initial_screenshot=True,
            # OpenAI supports these additional parameters
            reasoning_mode="auto",  # Enable reasoning summaries
            max_completion_tokens=4000,
        )

        try:
            print("🤖 OpenAI Agent Example")
            print("=" * 50)

            # Task that benefits from structured reasoning
            task = """
            I need you to:
            1. Go to https://www.timeanddate.com/worldclock/
            2. Find the current time in Tokyo, New York, and London
            3. Calculate the time differences between each pair of cities
            4. Take a screenshot showing all three times
            """

            print(f"📋 Task: Multi-city time comparison")
            print(f"🚀 Running OpenAI agent...\n")

            # Run the task
            result = await agent.run(task, max_steps=10)

            # OpenAI-specific: Access structured reasoning
            if hasattr(result, "reasoning_summary"):
                print(f"\n🧠 Reasoning Summary:")
                print(f"   {result.reasoning_summary}")

            print(f"\n✅ Task completed!")
            print(f"   Success: {result.success}")
            print(f"   Total steps: {len(result.tool_calls)}")

            # Show detailed step analysis
            print(f"\n📊 Step-by-step breakdown:")
            for i, call in enumerate(result.tool_calls, 1):
                action = "unknown"
                if hasattr(call, "arguments") and call.arguments:
                    action = call.arguments.get("action", "unknown")
                print(f"   Step {i}: {call.name} - {action}")

            # OpenAI models often provide structured data
            if result.final_answer:
                print(f"\n📝 Final answer: {result.final_answer}")

        finally:
            await client.close()

    print("\n✨ OpenAI agent demo complete!")
    print("\n💡 Tip: OpenAI models excel at tasks requiring structured reasoning")
    print("   and precise function calling sequences.")


if __name__ == "__main__":
    asyncio.run(main())
