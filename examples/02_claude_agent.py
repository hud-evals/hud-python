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
from hud.agents import ClaudeAgent
from hud.clients import MCPClient
from hud.settings import settings


async def main():
    with hud.trace("Claude Agent Demo"):
        # For any environment, you can run :
        # hud debug <IMAGE_NAME> to see the logs
        # hud analyze <IMAGE_NAME> to get a report about its capabilities (tools, resources, etc.)
        # e.g. hud analyze hudpython/hud-remote-browser:latest

        mcp_config = {
            "hud": {
                "url": "https://mcp.hud.ai/v3/mcp",
                "headers": {
                    "Authorization": f"Bearer {settings.api_key}",
                    "Mcp-Image": "hudpython/hud-remote-browser:latest",
                },
            }
        }

        # Create Claude-specific agent
        client = MCPClient(mcp_config=mcp_config)
        agent = ClaudeAgent.create(
            mcp_client=client,
            checkpoint_name="claude-sonnet-4-5",
            allowed_tools=["anthropic_computer"],
            initial_screenshot=True,
        )

        initial_url = "https://httpbin.org/forms/post"

        prompt = f"""
        Please help me test a web form:
        1. Navigate to {initial_url}
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

        await client.call_tool(
            name="setup",
            arguments={"name": "navigate_to_url", "arguments": {"url": initial_url}},
        )

        # Run the task
        await agent.run(prompt, max_steps=15)

    print("\n✨ Claude agent demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
