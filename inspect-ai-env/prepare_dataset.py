#!/usr/bin/env python3
"""Prepare inspect_ai dataset for use with Hud eval.

This script:
1. Loads an inspect_ai eval task (e.g., mbpp, swe_bench)
2. Analyzes its requirements (sandbox tools needed)
3. Converts each sample to Hud task format
4. Saves as JSONL with one task per line

Works with any inspect_ai eval.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add current directory to sys.path to enable importing local inspect_evals
if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

from inspect_loader import load_inspect_task
from task_converter import convert_and_save

OUTPUT_FILE = "samples.jsonl"


def install_eval_dependencies(eval_name: str) -> bool:
    """
    Install optional dependencies for an eval.

    Since inspect_evals is installed by cloning (not pip), we need to install
    dependencies directly.

    Args:
        eval_name: Base name of the eval (e.g., "swe_bench", "mbpp")

    Returns:
        True if dependencies were installed (requires restart), False otherwise
    """
    from importlib.util import find_spec

    print(f"   📦 Checking dependencies for '{eval_name}'...")

    # First check if dependencies are already available
    deps_needed = check_eval_dependencies(eval_name)

    if not deps_needed:
        print(f"   ✅ Dependencies already installed for '{eval_name}'")
        return False

    # Map eval names to their pip package requirements
    dependency_packages = {
        "swe_bench": ["swebench>=3.0.15", "docker"],
        "mathematics": ["sympy", "antlr4-python3-runtime==4.13.2"],
        "mle_bench": ["mlebench", "docker"],
        # Add more as needed
    }

    packages = dependency_packages.get(eval_name)
    if not packages:
        print(f"   ℹ️  No known dependencies for '{eval_name}'")
        return False

    print(f"   📦 Installing dependencies: {', '.join(packages)}...")
    deps_installed = False

    try:
        # Install packages directly
        result = subprocess.run(
            ["uv", "pip", "install"] + packages,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode == 0:
            print(f"   ✅ Installed dependencies for '{eval_name}'")
            deps_installed = True
        else:
            print(f"   ⚠️  Could not install dependencies: {result.stderr[:200]}")
            print(f"      Continuing anyway...")

    except subprocess.TimeoutExpired:
        print(f"   ⚠️  Dependency installation timed out")
    except Exception as e:
        print(f"   ⚠️  Dependency installation error: {e}")

    return deps_installed


def check_eval_dependencies(eval_name: str) -> bool:
    """
    Check if an eval's dependencies are installed by testing the actual import
    that the eval will use.

    Args:
        eval_name: Base name of the eval

    Returns:
        True if dependencies are needed but not installed, False otherwise
    """
    # For swe_bench, we need to check what the eval actually checks
    # Looking at the error: "assert find_spec("swebench")"
    # So we should check using importlib.util.find_spec

    from importlib.util import find_spec

    # Map of eval names to required import names
    dependency_map = {
        "swe_bench": "swebench",
        "mathematics": "sympy",
        "mle_bench": "mlebench",
        # Add more as needed
    }

    required_package = dependency_map.get(eval_name)
    if not required_package:
        # No known dependencies
        return False

    # Check if package is importable using find_spec (same as what evals use)
    try:
        spec = find_spec(required_package)
        if spec is None:
            return True  # Needs installation
        return False  # Already installed
    except (ImportError, ValueError, AttributeError):
        return True  # Needs installation


def download_eval_if_needed(eval_name: str) -> bool:
    """
    Download eval from inspect_evals repo if it's not already present,
    and install any required dependencies.

    Args:
        eval_name: Name of the eval (e.g., "mbpp", "swe_bench")

    Returns:
        True if dependencies were just installed (requires restart), False otherwise
    """
    # Only download if it looks like an official inspect eval (not custom_evals)
    if "custom_evals" in eval_name:
        return False

    # Extract the base eval name (e.g., "mbpp" from "mbpp" or "inspect_evals.mbpp")
    base_eval_name = eval_name
    if ":" in base_eval_name:
        base_eval_name = base_eval_name.split(":")[0]
    if "." in base_eval_name:
        base_eval_name = base_eval_name.split(".")[-1]

    # Check if already downloaded
    eval_dir = Path(f"inspect_evals/{base_eval_name}")
    already_downloaded = eval_dir.exists()

    if already_downloaded:
        print(f"   Eval '{base_eval_name}' already downloaded")
    else:
        # Try to download
        if not Path("download-eval.sh").exists():
            print(f"   ⚠️  download-eval.sh not found, skipping download")
            return False

        print(f"   📥 Downloading eval '{base_eval_name}'...")
        env = os.environ.copy()
        env["TARGET_EVAL"] = base_eval_name

        try:
            result = subprocess.run(
                ["./download-eval.sh"],
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                print(f"   ✅ Downloaded '{base_eval_name}'")
            else:
                print(f"   ⚠️  Download failed: {result.stderr}")
                print(f"      Continuing anyway (might be a custom eval)")
                return False  # Skip dependency install if download failed
        except Exception as e:
            print(f"   ⚠️  Download error: {e}")
            print(f"      Continuing anyway (might be a custom eval)")
            return False

    # Install dependencies (whether just downloaded or already present)
    return install_eval_dependencies(base_eval_name)


def prepare_dataset(
    eval_name: str,
    output_file: str = OUTPUT_FILE,
    task_params: dict | None = None,
    mcp_config: dict | None = None,
    limit: int | None = None,
) -> None:
    """
    Prepare inspect_ai dataset for use with Hud eval.

    Args:
        eval_name: Name of the eval (e.g., "mbpp", "inspect_evals.swe_bench:swe_bench")
        output_file: Path to output JSONL file
        task_params: Optional parameters to pass to the task function
        mcp_config: Optional MCP configuration (defaults to local docker)
        limit: Optional limit on number of samples to convert
    """
    print(f"\n📦 Preparing dataset for {eval_name}...")

    # Download eval if needed and install dependencies
    deps_installed = download_eval_if_needed(eval_name)
    if deps_installed:
        print(f"\n✅ Dependencies installed successfully!")
        print(f"⚠️  Please run the command again to use the newly installed packages:")
        print(
            f"    uv run python prepare_dataset.py --eval {eval_name} {f'--limit {limit}' if limit else ''}"
        )
        sys.exit(0)

    # Add default params for evals that need them
    if task_params is None:
        task_params = {}

    # For swe_bench, disable docker image building during dataset prep
    base_eval_name = eval_name.split(":")[0].split(".")[-1]
    if base_eval_name == "swe_bench":
        if "build_docker_images" not in task_params:
            task_params["build_docker_images"] = False
            print(f"   ℹ️  Setting build_docker_images=False for dataset preparation")

    # Set default model for inspect_ai if not already set
    # Some evals require a model during task loading for LLM-as-a-judge scoring
    # This is only used for task definition; actual scoring uses the agent's model
    if not os.getenv("INSPECT_EVAL_MODEL"):
        default_model = "openai/gpt-4o"
        os.environ["INSPECT_EVAL_MODEL"] = default_model
        print(f"   ℹ️  Set INSPECT_EVAL_MODEL={default_model} for task loading")
        print(f"      (Actual scoring will use your chosen agent model)")

    # Load eval task
    try:
        print(f"   Loading task...")
        task, requirements = load_inspect_task(eval_name, task_params)
        print(f"   Dataset size: {len(task.dataset)} samples")
        print(f"   Required tools: {requirements.get_required_tools()}")
        print(f"   Sandbox type: {requirements.sandbox_type}")
    except Exception as e:
        print(f"❌ Failed to load task: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Optionally limit samples
    if limit and limit < len(task.dataset):
        print(f"   Limiting to first {limit} samples")
        task.dataset = task.dataset[:limit]

    # Convert to Hud tasks
    try:
        print(f"   Converting to Hud task format...")
        hud_tasks = convert_and_save(
            task=task,
            requirements=requirements,
            eval_name=eval_name,
            output_path=output_file,
            mcp_config=mcp_config,
        )

        print(f"✅ Saved {len(hud_tasks)} tasks to {output_file}")
        print(f"\n💡 Usage:")
        print(f"   1. Start the sandbox: hud dev --build")
        print(f"   2. Run evaluation: hud eval {output_file} claude")

    except Exception as e:
        print(f"❌ Failed to convert tasks: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare inspect_ai eval dataset for use with Hud"
    )
    parser.add_argument(
        "--eval",
        type=str,
        help="Eval name (e.g., 'mbpp', 'inspect_evals.swe_bench:swe_bench'). "
        "If not provided, uses TARGET_EVAL environment variable.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_FILE,
        help=f"Output file (default: {OUTPUT_FILE})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of samples to convert (useful for testing)",
    )
    parser.add_argument(
        "--task-params",
        type=str,
        help="Task parameters as JSON string (e.g., '{\"temperature\": 0.5}')",
    )

    args = parser.parse_args()

    # Check if output file already exists
    if os.path.exists(args.output):
        print(
            f"❌ {args.output} already exists. Please remove it first or use --output to specify a different file."
        )
        sys.exit(1)

    # Get eval name
    eval_name = args.eval or os.getenv("TARGET_EVAL")
    if not eval_name:
        print(
            "❌ No eval specified. Use --eval or set TARGET_EVAL environment variable."
        )
        parser.print_help()
        sys.exit(1)

    # Parse task params if provided
    task_params = None
    if args.task_params:
        try:
            task_params = json.loads(args.task_params)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid task params JSON: {e}")
            sys.exit(1)

    # Prepare dataset (will auto-download if needed)
    prepare_dataset(
        eval_name=eval_name,
        output_file=args.output,
        task_params=task_params,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
