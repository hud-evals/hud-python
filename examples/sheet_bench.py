#!/usr/bin/env python3
"""
SheetBench Agent Example

Prerequisites:
- uv add hud-python
- Set HUD_API_KEY environment variable

Usage:
- Run single task: python sheet_bench.py
- Run entire dataset: python sheet_bench.py dataset
"""

import asyncio
import hud
from hud.agents import ClaudeAgent
from hud.clients import MCPClient
from datasets import load_dataset
from hud.datasets import run_dataset, TaskConfig


async def run_single_task():
    """Run a single task from SheetBench dataset."""
    # Load the dataset
    print("📊 Loading SheetBench dataset...")
    dataset = load_dataset("hud-evals/SheetBench-50", split="train")

    with hud.trace("SheetBench Agent"):
        task = TaskConfig(**dataset[0])

        # Create client and agent
        client = MCPClient(mcp_config=task.mcp_config)
        agent = ClaudeAgent(
            mcp_client=client,
            model="claude-sonnet-4-20250514",
            allowed_tools=["anthropic_computer"],
            initial_screenshot=True,
        )

        try:
            print(task.prompt)
            result = await agent.run(task, max_steps=40)
            print(result.reward)

        finally:
            print("\n🔚 Closing client...")
            await client.close()


async def run_sheetbench_dataset():
    """Run the entire SheetBench dataset using run_dataset."""
    # Load the dataset
    print("📊 Loading SheetBench dataset...")
    dataset = load_dataset("hud-evals/SheetBench-50", split="train")

    # Define agent configuration
    agent_config = {
        "model": "claude-sonnet-4-20250514",
        "allowed_tools": ["anthropic_computer"],
        "initial_screenshot": True,
    }

    # Run the dataset
    print("🚀 Running SheetBench dataset evaluation...")
    results = await run_dataset(
        name="SheetBench-50 Evaluation",
        dataset=dataset,
        agent_class=ClaudeAgent,
        agent_config=agent_config,
        max_concurrent=50,
        metadata={
            "dataset": "SheetBench-50",
            "split": "train",
        },
        max_steps=150,
    )

    # Process results
    print("\n📈 Results Summary:")
    rewards = [r.reward for r in results if r is not None]
    if rewards:
        print(f"  Total tasks: {len(results)}")
        print(f"  Successful tasks: {len(rewards)}")
        print(f"  Average reward: {sum(rewards) / len(rewards):.2f}")
        print(f"  Min reward: {min(rewards):.2f}")
        print(f"  Max reward: {max(rewards):.2f}")
    else:
        print("  No successful tasks completed.")

    return results


async def main():
    """Main entry point - choose which function to run."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "dataset":
        # Run the entire dataset
        await run_sheetbench_dataset()
    else:
        # Run a single task (default)
        await run_single_task()


if __name__ == "__main__":
    asyncio.run(main())
