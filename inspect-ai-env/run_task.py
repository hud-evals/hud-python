#!/usr/bin/env python3


from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
import traceback

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add current directory to sys.path to enable importing local inspect_evals
if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

from hud.clients import MCPClient


async def run_single_sample(
    eval_name: str, sample_dict: dict, task_params: dict = None, mcp_config: dict = None
) -> dict:
    """
    Run evaluation on a single sample.

    Args:
        eval_name: Name of the eval (e.g., "mbpp", "swe_bench")
        sample_dict: Sample data dict with keys: id, input, target, metadata, etc.
        task_params: Optional parameters for the eval's task function
        mcp_config: Optional MCP configuration

    This is designed for parallel processing where each Docker container
    processes a single sample from the eval's dataset.
    """
    if mcp_config is None:
        mcp_config = {"inspect_ai_env": {"url": "http://localhost:8765/mcp"}}

    client = MCPClient(mcp_config=mcp_config)

    try:
        print("🔧 Initializing MCP client...")
        await client.initialize()

        print(f"📋 Running setup for {eval_name}...")
        setup_result = await client.call_tool(
            name="setup", arguments={"eval_name": eval_name}
        )
        print(f"✅ Setup: {setup_result.content}")

        sample_id = sample_dict.get("id", "unknown")
        print(f"\n🔄 Running evaluation on sample: {sample_id}")
        print(f"   Eval: {eval_name}")
        if task_params:
            print(f"   Task params: {task_params}")

        result = await client.call_tool(
            name="evaluate",
            arguments={
                "eval_name": eval_name,
                "task_params": task_params or {},
                "sample": sample_dict,
            },
        )
        result = json.loads(result.content[0].text)
        print(f"\n📊 Results:\n{result}")

        if result.get("isError"):
            print(f"❌ Evaluation failed: {result.get('content')}")
            return {
                "sample_id": sample_id,
                "success": False,
                "error": result.get("content"),
            }

        print(f"✅ Evaluation complete!")

        return {
            "sample_id": sample_id,
            "success": True,
            "reward": result.get("reward"),
            "content": result.get("content"),
        }

    except Exception as e:
        print(f"❌ Exception during evaluation: {e}")
        if "connection" in str(e).lower():
            print("💡 Make sure 'hud dev --build' is running in another terminal")
        traceback.print_exc()
        return {
            "sample_id": sample_dict.get("id", "unknown"),
            "success": False,
            "error": str(e),
        }
    finally:
        await client.shutdown()


async def main():

    import argparse

    parser = argparse.ArgumentParser(
        description="Run inspect_ai evaluations with HUD integration"
    )
    parser.add_argument(
        "sample_id",
        type=str,
        help="Sample id to process",
    )

    args = parser.parse_args()

    # Load eval name from environment
    eval_name = os.getenv("TARGET_EVAL")
    if not eval_name:
        print("❌ TARGET_EVAL environment variable not set")
        sys.exit(1)

    # Parse task params
    with open("tasks.json", "r") as f:
        task_params = json.load(f)

    print("🚀 Inspect AI Evaluation with HUD Integration")
    print("=" * 60)
    print(f"📝 Eval: {eval_name}")

    if args.sample_id is None:
        print("❌ Must specify sample_index")
        parser.print_help()
        sys.exit(1)

    target_sample_dict = None
    with open("samples.jsonl", "r") as f:
        for sample in f:
            sample_dict = json.loads(sample)
            if sample_dict.get("id") == args.sample_id:
                target_sample_dict = sample_dict

    if target_sample_dict is None:
        print(f"❌ Could not find {args.sample_id} in samples.json")
        sys.exit(1)

    # Run single sample
    result = await run_single_sample(
        eval_name, target_sample_dict, task_params=task_params
    )

    # Exit with appropriate code
    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    asyncio.run(main())
