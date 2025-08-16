#!/usr/bin/env python3
"""
Text 2048 Example - Two ways to run the environment:

1. Direct Python (no Docker) - Runs in your current environment
2. Docker container - Simpler config but requires building the image first
"""

import asyncio
import sys
from pathlib import Path
import hud
from hud.datasets import TaskConfig
from hud.agents import ClaudeAgent
from hud.clients import MCPClient


async def main():
    with hud.trace("Hello 2048 Game"):
        # THERE ARE TWO WAYS TO RUN THE LOCAL ENVIRONMENT
        # OPTION 1: Direct Python execution (Not recommended for production use)
        text_2048 = Path(__file__).parent.parent / "environments/text_2048"
        mcp_config = {
            "local": {
                "command": sys.executable,
                "args": ["-m", "hud_controller.server"],
                "env": {"PYTHONPATH": str(text_2048 / "src")},
                "cwd": str(text_2048),
            }
        }

        # OPTION 2: Needs: docker build -t hud-text-2048 environments/text_2048
        # But this allows for running an arbitrary environment in the same way!
        # mcp_config = {
        #     "local": {
        #         "command": "docker",
        #         "args": ["run", "--rm", "-i", "hud-text-2048"]
        #     }
        # }

        task_dict = {
            "prompt": "Play 2048 and get the highest score possible.",
            "mcp_config": mcp_config,
            # Setup and evaluated tools are defined by the environment (see environments/text_2048/)
            "setup_tool": {
                "name": "setup",
                "arguments": {"name": "board", "arguments": {"board_size": 4}},
            },
            "evaluate_tool": {"name": "evaluate", "arguments": {"name": "max_number"}},
        }
        task = TaskConfig(**task_dict)

        # All of our environments use MCP as a generalizeable interface to interact with the environment
        client = MCPClient(mcp_config=task.mcp_config)

        # Define the agent that uses a VLM and can call tools via the client
        agent = ClaudeAgent(mcp_client=client, allowed_tools=["move"])

        try:
            # Running a full TaskConfig automatically handles setup and evaluation!
            result = await agent.run(task, max_steps=-1)
            print(f"Game completed! Reward: {result.reward}")
        finally:
            await client.close()


if __name__ == "__main__":
    asyncio.run(main())
