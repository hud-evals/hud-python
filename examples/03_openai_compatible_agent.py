#!/usr/bin/env python3
"""
OpenAI-compatible Chat Agent playing 2048 (text or browser).

Usage:
  python examples/openai_compatible_agent.py --mode text    # default
  python examples/openai_compatible_agent.py --mode browser

Requirements:
- pip install openai
- export OPENAI_API_KEY="your-api-key"  # Or set OPENAI_BASE_URL for custom endpoints

Environment Variables:
- OPENAI_BASE_URL: Custom OpenAI-compatible API endpoint
- OPENAI_API_KEY: API key for authentication
"""

import argparse
import asyncio
import os
from typing import Literal

from openai import AsyncOpenAI

import hud
from hud.agents.openai_chat import OpenAIChatAgent
from hud.datasets import LegacyTask


def _system_prompt(mode: Literal["text", "browser"]) -> str:
    if mode == "browser":
        return (
            "You are an expert 2048 game player using a browser interface. Your goal is to reach the tile specified by the user.\n\n"
            "HOW 2048 WORKS:\n"
            "- 4x4 grid with numbered tiles (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048...)\n"
            "- When you move, all tiles slide in that direction\n"
            "- When two tiles with SAME number touch, they merge into one (2+2=4, 4+4=8, etc.)\n"
            "- After each move, a new tile (2 or 4) appears randomly\n"
            "- Game ends when grid is full and no merges possible\n\n"
            "BROWSER INTERACTION USING THE COMPUTER TOOL:\n"
            '1. FIRST TURN ONLY - TAKE SCREENSHOT: Use: computer(action="screenshot")\n'
            "   After that, the environment returns an image with each successful move.\n"
            '2. MAKE MOVES - Use arrow keys with action="press": up/down/left/right.\n\n'
            "CRITICAL RULES:\n"
            "- Make exactly ONE move per turn using arrow keys\n"
            "- Continue until target or game ends; no confirmations needed.\n\n"
            "Strategy: keep highest tiles in a corner; maintain order; avoid random moves."
        )
    # text
    return (
        "You are an expert 2048 game player. Your goal is to reach the tile specified by the user.\n\n"
        "HOW 2048 WORKS:\n"
        "- 4x4 grid with numbered tiles (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048...)\n"
        "- When you move, all tiles slide in that direction\n"
        "- When two tiles with SAME number touch, they merge into one (2+2=4, 4+4=8, etc.)\n"
        "- After each move, a new tile (2 or 4) appears randomly\n"
        "- Game ends when grid is full and no merges possible\n\n"
        "CRITICAL RULES:\n"
        "- ALWAYS analyze the board before moving\n"
        "- ALWAYS make a tool call for your move\n"
        "- Use the 'move' tool with these choices: up, down, left, right\n"
        "- Make exactly ONE move per turn\n"
        "- NEVER ask for permission; just keep playing until the game ends\n"
        "- Don't ask 'Should I continue?'; just make your next move\n\n"
        'Example tool call: {"name": "move", "arguments": {"direction": "right"}}'
    )


def _task_for_mode(mode: Literal["text", "browser"], target: int) -> LegacyTask:
    if mode == "browser":
        mcp_config = {
            "local": {
                "command": "docker",
                "args": ["run", "--rm", "-i", "-p", "8080:8080", "hudevals/hud-browser:0.1.3"],
            }
        }
        prompt = (
            "Play the browser-based 2048 game and try to reach the target tile. "
            "Start by taking a screenshot, then make strategic moves using arrow keys."
        )
        setup_tool = {"name": "launch_app", "arguments": {"app_name": "2048"}}
        evaluate_tool = {
            "name": "evaluate",
            "arguments": {"name": "game_2048_max_number", "arguments": {"target": target}},
        }
    else:
        mcp_config = {
            "local": {
                "command": "docker",
                "args": ["run", "--rm", "-i", "hudevals/hud-text-2048:0.1.6"],
            }
        }
        prompt = f"Aim for the {target} tile (at least a score of 800!)"
        setup_tool = {
            "name": "setup",
            "arguments": {"name": "board", "arguments": {"board_size": 4}},
        }
        evaluate_tool = {
            "name": "evaluate",
            "arguments": {"name": "max_number", "arguments": {"target": target}},
        }

    return LegacyTask(
        prompt=prompt,
        mcp_config=mcp_config,
        setup_tool=setup_tool,  # type: ignore[arg-type]
        evaluate_tool=evaluate_tool,  # type: ignore[arg-type]
    )


async def run_example(mode: Literal["text", "browser"], target: int) -> None:
    # Initialize OpenAI client with environment variables
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")

    openai_client = AsyncOpenAI(
        base_url=base_url if base_url else None,
        api_key=api_key,
    )

    task = _task_for_mode(mode, target)
    system_prompt = _system_prompt(mode)

    checkpoint = "gpt-5-mini"  # Replace with your model checkpoint

    # Allowed tools differ by mode
    allowed_tools = ["computer"] if mode == "browser" else ["move"]

    # Create OpenAI-compatible agent
    agent = OpenAIChatAgent.create(
        openai_client=openai_client,
        checkpoint_name=checkpoint,
        allowed_tools=allowed_tools,
        append_setup_output=False,
        system_prompt=system_prompt,
    )

    title = "OpenAI 2048 Game (Browser)" if mode == "browser" else "OpenAI 2048 Game (Text)"
    print("🎮 Starting 2048 game with OpenAI-compatible agent...")
    print(f"🤖 Model: {agent.config.checkpoint_name}")
    print(f"🧩 Mode: {mode}")
    print("=" * 50)

    # Use hud.eval() with Task.from_v4() for legacy task format
    from hud.eval.task import Task

    v5_task = Task.from_v4(task)
    async with hud.eval(v5_task, variants={"model": checkpoint, "mode": mode}) as ctx:
        result = await agent.run(ctx, max_steps=100)

    print("=" * 50)
    print("✅ Game completed!")
    print(f"🏆 Final Score/Max Tile: {result.reward}")
    if result.info:
        print(f"📊 Game Stats: {result.info}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenAI-compatible 2048 example")
    p.add_argument(
        "--mode", choices=["text", "browser"], default="text", help="Which environment to run"
    )
    p.add_argument("--target", type=int, default=128, help="Target tile")
    return p.parse_args()


async def main() -> None:
    args = _parse_args()
    await run_example(args.mode, args.target)


if __name__ == "__main__":
    asyncio.run(main())
