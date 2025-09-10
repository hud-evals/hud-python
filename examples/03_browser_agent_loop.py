#!/usr/bin/env python3
"""
Browser Environment Example - Simple version like 01_hello_2048.py

This example demonstrates browser automation with HUD:
- Configurable for different apps (2048, todo)
- Simple task definition with setup and evaluation
- Agent automatically handles the full lifecycle

Usage:
    python 03_browser_agent_loop.py          # defaults to 2048
    python 03_browser_agent_loop.py --app todo
"""

import asyncio
import argparse
import hud
from hud.datasets import Task
from hud.agents import ClaudeAgent
from hud.clients import MCPClient


async def main():
    parser = argparse.ArgumentParser(description="Browser Environment Example")
    parser.add_argument(
        "--app",
        type=str,
        default="2048",
        choices=["2048", "todo"],
        help="Which app to run (2048 or todo)",
    )

    args = parser.parse_args()

    print(f"🎮 Browser Environment Example - {args.app.upper()}")
    print("=" * 50)

    # MCP configuration for browser environment
    mcp_config = {
        "local": {
            "command": "docker",
            "args": ["run", "--rm", "-i", "-p", "8080:8080", "hudevals/hud-browser:0.1.3"],
        }
    }

    # Define tasks for each app
    if args.app == "2048":
        task_dict = {
            "prompt": """Play the 2048 game and try to reach the 512 tile.
            
            Strategy tips:
            - Keep your highest tiles in a corner
            - Build tiles in descending order
            - Avoid random moves
            - Use arrow keys or swipe to move tiles
            
            Make strategic moves to maximize your score and reach the target.""",
            "mcp_config": mcp_config,
            "setup_tool": {
                "name": "setup",
                "arguments": [
                    {"name": "launch_app", "arguments": {"app_name": "2048"}},
                    {"name": "game_2048_board", "arguments": {"board_size": 4, "target_tile": 512}},
                ],
            },
            "evaluate_tool": {
                "name": "evaluate",
                "arguments": {"name": "game_2048_max_number", "arguments": {"target": 512}},
            },
        }
    else:  # todo app
        task_dict = {
            "prompt": """Complete the following tasks in the todo app:
            
            1. Mark 2 more items as completed (you'll see 5 items total)
            2. Use the filter features to view completed/active items
            3. Test the search functionality
            
            Be systematic and interact with the app's features.""",
            "mcp_config": mcp_config,
            "setup_tool": {
                "name": "setup",
                "arguments": [
                    {"name": "launch_app", "arguments": {"app_name": "todo"}},
                    {"name": "todo_seed", "arguments": {"num_items": 5}},
                ],
            },
            "evaluate_tool": {
                "name": "evaluate",
                "arguments": {"name": "todo_completed", "arguments": {"expected_count": 4}},
            },
        }

    task = Task(**task_dict)

    # Create MCP client
    client = MCPClient(mcp_config=task.mcp_config)

    # Create agent with browser automation capabilities
    agent = ClaudeAgent(
        mcp_client=client,
        allowed_tools=["anthropic_computer"],
    )

    with hud.trace(f"Browser {args.app} Example"):
        try:
            # Running a full Task automatically handles setup, launch, and evaluation!
            result = await agent.run(task, max_steps=50)
            print(f"\n✅ Task completed! Final reward: {result.reward}")
        finally:
            await client.shutdown()

    print(f"\n✨ Browser environment example complete!")


if __name__ == "__main__":
    asyncio.run(main())
