#!/usr/bin/env python3
"""
Claude Agent Example

This example showcases Claude-specific features:
- Initial screenshot capture
- Thinking/reasoning display
- Computer tool usage
- Model-specific parameters

Claude is particularly good at visual understanding and
multi-step reasoning tasks.
"""

import asyncio
import hud
from hud.mcp import ClaudeMCPAgent
from hud.mcp.client import MCPClient
from datasets import load_dataset


async def main():
    # Load the dataset
    dataset = load_dataset("hud-evals/sheetbench-taskconfigs")

    with hud.trace("Claude Agent Demo"):
        mcp_config = {
            "hud": {
                "url": "https://mcp.hud.so/v3/mcp",
                "headers": {
                    "Authorization": "Bearer ${HUD_API_KEY}",
                    "Mcp-Image": "hudpython/hud-remote-browser:latest",
                    "Run-Id": "${RUN_ID}",
                },
            }
        }

        # Create Claude-specific agent
        client = MCPClient(mcp_config=mcp_config)
        agent = ClaudeMCPAgent(
            mcp_client=client,
            model="claude-3-7-sonnet-20250219",
            allowed_tools=["anthropic_computer"],
            initial_screenshot=True,
        )

        try:
            print("🤖 Claude Agent Example")
            print("=" * 50)

            # Complex multi-step task that benefits from Claude's reasoning
            task = """
            Please help me test a web form:
            1. Navigate to https://httpbin.org/forms/post
            2. Fill in the customer name as "Claude Test"
            3. Enter the telephone as "555-0123"
            4. Type "Testing form submission with Claude" in the comments
            5. Select a small pizza size
            6. Choose "bacon" as a topping
            7. Set delivery time to "20:30"
            8. Submit the form
            9. Verify the submission was successful
            """

            print(f"📋 Task: Multi-step form interaction")
            print(f"🚀 Running Claude agent...\n")

            # Run the task
            result = await agent.run(task, max_steps=15)

            # Claude-specific: Access thinking/reasoning if available
            if hasattr(result, "messages"):
                for msg in result.messages:
                    if hasattr(msg, "content") and isinstance(msg.content, str):
                        if "thinking:" in msg.content.lower():
                            print(f"\n💭 Claude's reasoning: {msg.content}")

            print(f"\n✅ Task completed!")
            print(f"   Success: {result.done}")
            print(f"   Total steps: {len(result.tool_calls)}")

            # Show tool usage summary
            tool_summary = {}
            for call in result.tool_calls:
                tool_name = call.name
                tool_summary[tool_name] = tool_summary.get(tool_name, 0) + 1

            print(f"\n📊 Tool usage:")
            for tool, count in tool_summary.items():
                print(f"   - {tool}: {count} calls")

        finally:
            await client.close()

    print("\n✨ Claude agent demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
