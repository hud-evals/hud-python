#!/usr/bin/env python3
"""
SheetBench Agent Example

This example showcases SheetBench-specific features:
- Initial screenshot capture
- Thinking/reasoning display
- Computer tool usage
- Model-specific parameters

SheetBench
"""

import asyncio
import hud
from hud.mcp import ClaudeMCPAgent
from hud.mcp.client import MCPClient
from datasets import load_dataset
from hud.datasets import to_taskconfigs


async def main():
    # Load the dataset
    dataset = load_dataset("hud-evals/sheetbench-taskconfigs")
    with hud.trace("Claude Agent Demo"):
        tsx = to_taskconfigs(dataset["train"])
        task = tsx[0]

        client = MCPClient(mcp_config=task.mcp_config)
        agent = ClaudeMCPAgent(
            mcp_client=client,
            model="claude-3-7-sonnet-20250219",
            allowed_tools=["anthropic_computer"],
            initial_screenshot=True,
        )

        try:
            print(task.prompt)
            result = await agent.run(task, max_steps=15)
            print(result.reward)

        finally:
            await client.close()

    print("\n✨ SheetBench agent demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
