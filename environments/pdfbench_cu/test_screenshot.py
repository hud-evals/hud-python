#!/usr/bin/env python3
"""Simple test to verify PDF loading and screenshot in pdfbench_cu."""

import asyncio
from hud.clients import MCPClient


async def main():
    print("Creating MCP client...")

    config = {
        "local": {
            "command": "docker",
            "args": [
                "run", "--rm", "-i",
                "-v", "/Users/erikq/Documents/bytedance/harbor-pdfbench/harbor_tasks/pdfbench_eyemed_x001/environment/pdfs:/app/pdfs:ro",
                "-v", "/Users/erikq/Documents/bytedance/harbor-pdfbench/harbor_tasks/pdfbench_eyemed_x001/environment/solution.json:/opt/tbench/solution.json:ro",
                "-e", "PDF_PATH=/app/pdfs/eyemed.pdf",
                "-e", "SOLUTION_PATH=/opt/tbench/solution.json",
                "hud-pdfbench-cu:latest"
            ]
        }
    }

    client = MCPClient(mcp_config=config)
    await client.initialize()

    # List tools
    tools = await client.list_tools()
    print(f"Tools available: {[t.name for t in tools]}")

    # Take screenshot via computer tool
    print("\nTaking screenshot...")
    result = await client.call_tool(name="computer", arguments={"action": "screenshot"})

    # Check result
    if hasattr(result, "content") and result.content:
        for item in result.content:
            if hasattr(item, "data") and item.data:
                print(f"Got base64 image! Length: {len(item.data)} chars")
                # Save first 100 chars to verify it's base64
                print(f"First 100 chars: {item.data[:100]}...")
            elif hasattr(item, "text"):
                print(f"Text response: {item.text[:500] if len(item.text) > 500 else item.text}")
    else:
        print(f"Raw result: {result}")

    await client.shutdown()
    print("\nTest completed!")


if __name__ == "__main__":
    asyncio.run(main())
