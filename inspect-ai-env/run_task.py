#!/usr/bin/env python3


from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add current directory to sys.path to enable importing local inspect_evals
if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

from hud.clients import MCPClient


def load_eval_dataset(eval_name: str, task_params: dict = None):
    """
    Load an eval's dataset to extract samples.

    Supports both official inspect_evals and custom evals.

    Args:
        eval_name: Can be:
            - Simple name: "mbpp" → loads from inspect_evals.mbpp
            - Module path: "custom_evals.my_eval" → loads from that path
            - With function: "custom_evals.my_eval:my_task" → explicit function

    Returns:
        Dataset from the loaded task
    """
    from importlib import import_module

    try:
        # Parse eval_name
        if ":" in eval_name:
            module_path, function_name = eval_name.split(":", 1)
        else:
            module_path = eval_name
            function_name = None

        # Determine full module path
        if "." in module_path:
            # Custom eval with dots: "custom_evals.my_eval"
            full_module_path = module_path
            if not function_name:
                function_name = module_path.split(".")[-1]
        else:
            # Simple name: "mbpp" → "inspect_evals.mbpp"
            full_module_path = f"inspect_evals.{module_path}"
            if not function_name:
                function_name = module_path

        # Import and get task function
        eval_module = import_module(full_module_path)
        task_fn = getattr(eval_module, function_name)
        task = task_fn(**(task_params or {}))
        return task.dataset

    except ImportError as e:
        raise ValueError(
            f"Could not import eval '{eval_name}'. "
            f"For custom evals, ensure the module is accessible. Error: {e}"
        )
    except AttributeError as e:
        raise ValueError(
            f"Eval '{eval_name}' does not have function '{function_name}': {e}"
        )


def sample_to_dict(sample) -> dict:
    """Convert inspect_ai Sample object to dict for JSON serialization."""
    return {
        "id": sample.id,
        "input": str(sample.input) if sample.input else None,
        "target": sample.target,
        "metadata": sample.metadata or {},
        "sandbox": sample.sandbox,
    }


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

        if result.isError:
            print(f"❌ Evaluation failed: {result.content}")
            return {"sample_id": sample_id, "success": False, "error": result.content}

        print(f"✅ Evaluation complete!")
        print(f"\n📊 Results:\n{result.content}")

        return {
            "sample_id": sample_id,
            "success": True,
            "reward": result.reward,
            "content": result.content,
        }

    except Exception as e:
        print(f"❌ Exception during evaluation: {e}")
        if "connection" in str(e).lower():
            print("💡 Make sure 'hud dev --build' is running in another terminal")
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
        "sample_index",
        type=int,
        help="Sample index to process",
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

    if args.sample_index is not None:
        print("\n📦 Loading eval dataset...")
        try:
            dataset = load_eval_dataset(eval_name, task_params)
            print(f"   Dataset size: {len(dataset)} samples")

            if args.sample_index < 0 or args.sample_index >= len(dataset):
                print(
                    f"❌ Sample index {args.sample_index} out of range (dataset has {len(dataset)} samples)"
                )
                sys.exit(1)

            sample = dataset[args.sample_index]
            sample_dict = sample_to_dict(sample)
            print(f"   Sample ID: {sample_dict['id']}")

        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            sys.exit(1)

        # Run single sample
        result = await run_single_sample(
            eval_name, sample_dict, task_params=task_params
        )

    else:
        print("❌ Must specify sample_index")
        parser.print_help()
        sys.exit(1)

    # Exit with appropriate code
    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    asyncio.run(main())
