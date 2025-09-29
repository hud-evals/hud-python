#!/usr/bin/env python3
"""Prepare inspect_ai dataset for use with hud eval.

Downloads the eval dataset and converts each sample to HUD Task format,
saving as JSONL with one task per line.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

MCP_CONFIG = """{"hud": {"url": "https://mcp.hud.so/v3/mcp", "headers": {"Authorization": "Bearer ${HUD_API_KEY}", "Mcp-Image": "hudevals/hud-remote-browser:0.1.1"}}}"""
OUTPUT_FILE = "samples.jsonl"

# Add current directory to sys.path to enable importing local inspect_evals
if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))


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


def prepare_dataset(eval_name: str, hud_api_key: str) -> None:
    """
    Prepare inspect_ai dataset for use with hud eval.

    Downloads the eval dataset and converts each sample to HUD Task format,
    saving as JSONL with one task per line.

    Args:
        eval_name: Name of the eval (e.g., "mbpp", "swe_bench")
        task_params: Optional parameters for the eval's task function
        OUTPUT_FILE: Output JSONL file path
        mcp_url: MCP server URL for the tasks
    """
    print(f"\n📦 Preparing dataset for {eval_name}...")

    # Load eval dataset
    try:
        dataset = load_eval_dataset(eval_name, task_params)
        print(f"   Dataset size: {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        sys.exit(1)

    # Convert samples to HUD Task format
    tasks = []
    for i, sample in enumerate(dataset):
        sample_dict = sample_to_dict(sample)

        # Create HUD Task format
        task = {
            "id": f"{eval_name}_{sample_dict.get('id', i)}",
            "prompt": sample_dict.get("input", ""),
            "mcp_config": MCP_CONFIG.format(HUD_API_KEY=hud_api_key),
            "setup_tool": {"name": "setup", "arguments": {"eval_name": eval_name}},
            "evaluate_tool": {
                "name": "evaluate",
                "arguments": {
                    "eval_name": eval_name,
                    "task_params": task_params or {},
                    "sample": sample_dict,
                },
            },
            "metadata": {
                "eval_name": eval_name,
                "sample_id": sample_dict.get("id"),
                "target": sample_dict.get("target"),
            },
        }
        tasks.append(task)

    # Write to JSONL file
    with open(OUTPUT_FILE, "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")

    print(f"✅ Saved {len(tasks)} tasks to {OUTPUT_FILE}")
    print(f"\n💡 Usage: hud eval {OUTPUT_FILE} --full")


def main():
    # Check if output file already exists

    if os.path.exists(OUTPUT_FILE):
        print(f"❌ {OUTPUT_FILE} already exists. Please remove it first.")
        sys.exit(1)

    # Get eval name from environment
    eval_name = os.getenv("TARGET_EVAL")
    if not eval_name:
        print("❌ TARGET_EVAL not set in .env file")
        sys.exit(1)

    # Get eval name from environment
    hud_api_key = os.getenv("HUD_API_KEY")
    if not hud_api_key:
        print("❌ HUD_API_KEY not set in .env file")
        sys.exit(1)

    # Prepare dataset
    prepare_dataset(eval_name, hud_api_key)


if __name__ == "__main__":
    main()
