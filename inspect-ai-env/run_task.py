#!/usr/bin/env python3
"""
Inspect AI Single Sample Evaluation Runner

This script processes a SINGLE sample from an inspect_ai evaluation.
It's designed for parallel processing where each Docker container
handles one sample from the eval's dataset.

Architecture:
  1. Load eval to get dataset
  2. Extract specific sample by index
  3. Pass sample data into Docker container
  4. Container runs inspect_ai evaluation on that one sample
  5. Native solver/scorer from inspect_ai are used
  6. HUDAgentModel routes LLM calls to AGENT_CALLBACK_URL

Usage:
    # Process single sample by index
    python run_task.py mbpp 0

    # With task params
    python run_task.py mbpp 0 --task-params '{"temperature": 0.5}'

    # Batch mode (multiple samples, no parallelization)
    python run_task.py mbpp --limit 3
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

from hud.clients import MCPClient


def load_eval_dataset(eval_name: str, task_params: dict = None):
    """Load an eval's dataset to extract samples."""
    from importlib import import_module

    try:
        eval_module = import_module(f"inspect_evals.{eval_name}")
        task_fn = getattr(eval_module, eval_name)
        task = task_fn(**(task_params or {}))
        return task.dataset
    except ImportError as e:
        raise ValueError(f"Could not import eval '{eval_name}': {e}")
    except AttributeError as e:
        raise ValueError(f"Eval '{eval_name}' does not have a task function: {e}")


def sample_to_dict(sample) -> dict:
    """Convert inspect_ai Sample object to dict for JSON serialization."""
    return {
        "id": sample.id,
        "input": str(sample.input) if sample.input else None,
        "target": sample.target,
        "metadata": sample.metadata or {},
        "sandbox": sample.sandbox
    }


async def run_single_sample(
    eval_name: str,
    sample_dict: dict,
    task_params: dict = None,
    mcp_config: dict = None
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
        mcp_config = {
            "inspect_ai_env": {
                "url": "http://localhost:8765/mcp"
            }
        }

    client = MCPClient(mcp_config=mcp_config)

    try:
        print("🔧 Initializing MCP client...")
        await client.initialize()

        print(f"📋 Running setup for {eval_name}...")
        setup_result = await client.call_tool(
            name="setup",
            arguments={"eval_name": eval_name}
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
                "sample": sample_dict
            }
        )

        if result.isError:
            print(f"❌ Evaluation failed: {result.content}")
            return {
                "sample_id": sample_id,
                "success": False,
                "error": result.content
            }

        print(f"✅ Evaluation complete!")
        print(f"\n📊 Results:\n{result.content}")

        return {
            "sample_id": sample_id,
            "success": True,
            "reward": result.reward,
            "content": result.content
        }

    except Exception as e:
        print(f"❌ Exception during evaluation: {e}")
        if "connection" in str(e).lower():
            print("💡 Make sure 'hud dev --build' is running in another terminal")
        return {
            "sample_id": sample_dict.get("id", "unknown"),
            "success": False,
            "error": str(e)
        }
    finally:
        await client.shutdown()


async def run_batch(
    eval_name: str,
    task_params: dict = None,
    limit: int = None,
    mcp_config: dict = None
) -> dict:
    """
    Run evaluation on multiple samples (batch mode, no parallelization).

    For production parallel processing, use run_single_sample() instead
    and distribute samples across containers externally.
    """
    if mcp_config is None:
        mcp_config = {
            "inspect_ai_env": {
                "url": "http://localhost:8765/mcp"
            }
        }

    client = MCPClient(mcp_config=mcp_config)

    try:
        print("🔧 Initializing MCP client...")
        await client.initialize()

        print(f"📋 Running setup for {eval_name}...")
        setup_result = await client.call_tool(
            name="setup",
            arguments={"eval_name": eval_name}
        )
        print(f"✅ Setup: {setup_result.content}")

        print(f"\n🔄 Running evaluation: {eval_name}")
        if limit:
            print(f"   Limit: {limit} samples")
        if task_params:
            print(f"   Task params: {task_params}")

        result = await client.call_tool(
            name="evaluate",
            arguments={
                "eval_name": eval_name,
                "task_params": task_params or {},
                "limit": limit
            }
        )

        if result.isError:
            print(f"❌ Evaluation failed: {result.content}")
            return {
                "success": False,
                "error": result.content
            }

        print(f"✅ Evaluation complete!")
        print(f"\n📊 Results:\n{result.content}")

        return {
            "success": True,
            "reward": result.reward,
            "content": result.content
        }

    except Exception as e:
        print(f"❌ Exception during evaluation: {e}")
        if "connection" in str(e).lower():
            print("💡 Make sure 'hud dev --build' is running in another terminal")
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        await client.shutdown()


async def main():
    """
    Main function for running inspect_ai evaluations.

    Usage:
        # Single sample mode (for parallel processing)
        python run_task.py mbpp 0                    # Process sample at index 0
        python run_task.py mbpp 42 --task-params '{...}'

        # Batch mode (multiple samples, sequential)
        python run_task.py mbpp --limit 3
        python run_task.py swe_bench --limit 1 --task-params '{"dataset": "..."}'
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Run inspect_ai evaluations with HUD integration"
    )
    parser.add_argument("eval_name", help="Name of eval (e.g., mbpp, swe_bench, gpqa)")
    parser.add_argument("sample_index", nargs="?", type=int, help="Sample index to process (for single-sample mode)")
    parser.add_argument("--limit", type=int, help="Limit number of samples (batch mode)")
    parser.add_argument("--task-params", type=str, help="JSON string of task parameters")
    parser.add_argument("--output", help="Output file for results (default: stdout)")

    args = parser.parse_args()

    # Parse task params
    task_params = None
    if args.task_params:
        try:
            task_params = json.loads(args.task_params)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in --task-params: {e}")
            sys.exit(1)

    print("🚀 Inspect AI Evaluation with HUD Integration")
    print("=" * 60)
    print(f"📝 Eval: {args.eval_name}")
    if task_params:
        print(f"⚙️  Task params: {task_params}")

    # Determine mode: single sample or batch
    if args.sample_index is not None:
        # Single sample mode - load dataset and extract sample
        print(f"🎯 Mode: Single sample (index {args.sample_index})")
        print("=" * 60)

        print("\n📦 Loading eval dataset...")
        try:
            dataset = load_eval_dataset(args.eval_name, task_params)
            print(f"   Dataset size: {len(dataset)} samples")

            if args.sample_index < 0 or args.sample_index >= len(dataset):
                print(f"❌ Sample index {args.sample_index} out of range (dataset has {len(dataset)} samples)")
                sys.exit(1)

            sample = dataset[args.sample_index]
            sample_dict = sample_to_dict(sample)
            print(f"   Sample ID: {sample_dict['id']}")

        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            sys.exit(1)

        # Run single sample
        result = await run_single_sample(
            args.eval_name,
            sample_dict,
            task_params=task_params
        )

    elif args.limit:
        # Batch mode
        print(f"📦 Mode: Batch ({args.limit} samples)")
        print("=" * 60)

        result = await run_batch(
            args.eval_name,
            task_params=task_params,
            limit=args.limit
        )

    else:
        print("❌ Must specify either sample_index or --limit")
        parser.print_help()
        sys.exit(1)

    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n📄 Results saved to {args.output}")

    # Exit with appropriate code
    sys.exit(0 if result.get('success') else 1)


if __name__ == "__main__":
    asyncio.run(main())