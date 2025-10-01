#!/usr/bin/env python
"""
Simple example of running tasks from tasks.json. Make sure to have run hud build.
"""

from __future__ import annotations

import asyncio
import json

from hud.clients import MCPClient
from hud.datasets import Task


async def run_task(task_data: dict):
    task = Task(**task_data)
    client = MCPClient(mcp_config=task.mcp_config)

    try:
        print("Initializing client...")
        await client.initialize()

        result = await client.call_tool(task.setup_tool)  # type: ignore
        print(f"✅ Setup: {result.content}")

        print("\n🔄 Performing actions:")
        for _ in range(10):
            result = await client.call_tool(name="act", arguments={})
            print(f"  {result.content}")

        result = await client.call_tool(task.evaluate_tool)  # type: ignore
        print(f"\n📊 Evaluation: {result.content}")

        return result.content
    except Exception as e:
        if "connection" in str(e).lower():
            print(
                "❌ Could not connect. Make sure 'hud dev --build' is running in another terminal."
            )
        else:
            raise e
    finally:
        await client.shutdown()


async def main():
    for task_data in json.load(open("tasks.json")):
        await run_task(task_data)


if __name__ == "__main__":
    asyncio.run(main())
