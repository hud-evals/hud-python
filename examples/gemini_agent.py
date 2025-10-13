#!/usr/bin/env python3
"""
Gemini Agent Example (Remote Browser)

This example showcases Gemini-specific features against a remote browser environment:
- Computer use capabilities with normalized coordinates
- Browser automation
- Multi-step reasoning tasks

Gemini uses a normalized coordinate system (0-999) that is automatically
scaled to actual screen dimensions.
"""

import asyncio

import hud
import os
from hud.agents import GeminiAgent
from hud.clients import MCPClient
from hud.settings import settings


async def main():
    with hud.trace("Gemini Agent Demo"):
        # Remote HUD MCP server using your custom remote-browser image
        # Built via environments/remote_browser/Dockerfile
        # Build headers with required environment for remote browser
        provider = os.getenv("BROWSER_PROVIDER", "anchorbrowser")
        headers = {
            "Authorization": f"Bearer {settings.api_key}",
            "Mcp-Image": "alberthu233/hud-remote-browser:gemini-dev-1",
            "Env-Browser-Provider": provider,
        }

        # Optionally pass provider-specific API key if available
        provider_key_map = {
            "anchorbrowser": "ANCHOR_API_KEY",
            "steel": "STEEL_API_KEY",
            "browserbase": "BROWSERBASE_API_KEY",
            "hyperbrowser": "HYPERBROWSER_API_KEY",
            "kernel": "KERNEL_API_KEY",
        }
        if provider in provider_key_map:
            key_var = provider_key_map[provider]
            key_val = os.getenv(key_var)
            if key_val:
                header_key = f"Env-{'-'.join(part.capitalize() for part in key_var.split('_'))}"
                headers[header_key] = key_val

        mcp_config = {"hud": {"url": "https://mcp.hud.so/v3/mcp", "headers": headers}}

        # Create Gemini-specific agent
        client = MCPClient(mcp_config=mcp_config)
        agent = GeminiAgent(
            mcp_client=client,
            model="gemini-2.5-computer-use-preview-10-2025",
            allowed_tools=["gemini_computer"],
            initial_screenshot=True,
            temperature=1.0,
            max_output_tokens=8192,
        )

        await client.initialize()

        try:
            initial_url = "https://httpbin.org/forms/post"

            prompt = f"""
            Please help me fill out a web form one step at a time:
            1. Navigate to {initial_url}
            2. Fill in the customer name as "Gemini Test"
            3. Enter the telephone as "555-0456"
            4. Enter the email as "gemini@test.com"
            5. Type "Testing form submission with Gemini" in the comments
            6. Select a medium pizza size
            7. Choose "mushroom" as a topping
            8. Set delivery time to "10:00 AM"
            9. Submit the form
            10. Verify the submission was successful
            """

            print("📋 Task: Multi-step form interaction (Remote Browser)")
            print("🚀 Running Gemini agent...\n")

            # Setup: navigate to initial URL via setup tool
            await client.call_tool(
                name="setup",
                arguments={"name": "navigate_to_url", "arguments": {"url": initial_url}},
            )

            # Run the prompt
            result = await agent.run(prompt, max_steps=50)

            print(result)

        finally:
            await client.shutdown()

    print("\n✨ Gemini agent demo complete!")


if __name__ == "__main__":
    asyncio.run(main())

