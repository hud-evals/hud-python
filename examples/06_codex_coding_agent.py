#!/usr/bin/env python3
"""
Build Your Own Codex - A 1:1 Recreation of OpenAI's Codex CLI

This example shows how to build your own Codex (https://github.com/openai/codex)
from scratch using the HUD SDK. The implementation matches Codex's behavior
exactly because HUD's tools conform to the same OpenAI Responses API specs:

- `ShellTool` implements `ShellAction` → `ShellResult` (stdout, stderr, outcome)
- `ApplyPatchTool` implements V4A diff format (create_file, update_file, delete_file)

The `OpenAIAgent` automatically converts these to OpenAI's native tool types,
so the model sees the exact same interface as the official Codex CLI.

What you get:
- **Your own Codex** - Same behavior as `codex` CLI, but fully customizable
- **Full observability** - Every tool call and response traced on hud.ai
- **Two modes** - Local (like `codex`) or Hub (cloud sandboxed execution)

Usage:
  # Local mode - just like running `codex` on your machine
  uv run python examples/06_codex_coding_agent.py --local

  # Hub mode - sandboxed cloud execution with full telemetry
  export HUD_API_KEY="sk-hud-..."
  uv run python examples/06_codex_coding_agent.py

  # Custom task
  uv run python examples/06_codex_coding_agent.py --local \\
    --task "Create a Python script that prints the Fibonacci sequence"

Requirements:
  - Install deps: `uv sync`
  - HUD_API_KEY environment variable (for both local and hub modes)
"""

import argparse
import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load .env file from current directory or parent directories
load_dotenv()

import hud
from hud.agents.openai import OpenAIAgent
from hud.settings import settings
from hud.tools.coding import ApplyPatchTool, ShellTool

# =============================================================================
# Configuration
# =============================================================================

# Default hub environment name
DEFAULT_HUB = "codex_environment_sandbox"

# Codex-capable models that support native shell/apply_patch tools
CODEX_MODELS = {
    "gpt-5.1-codex",
    "gpt-5.1",
}


# =============================================================================
# Run Coding Task Locally (No Docker)
# =============================================================================


async def run_coding_task_local(
    task: str,
    model: str = "gpt-5.1",
    max_steps: int = 20,
    verbose: bool = False,
    work_dir: str | None = None,
) -> None:
    """
    Run a coding task locally without Docker.

    Uses ShellTool and ApplyPatchTool running on your local machine.
    Files are created in a temporary directory (or specified work_dir).

    Args:
        task: Description of the coding task
        model: OpenAI model to use (default: gpt-5.1)
        max_steps: Maximum agent steps (default: 20)
        verbose: Enable verbose output
        work_dir: Working directory for file operations (default: temp dir)
    """
    # Validate model is Codex-capable
    if model not in CODEX_MODELS:
        raise ValueError(
            f"Model '{model}' is not in the Codex-capable list {sorted(CODEX_MODELS)}.\n"
            "Use a model that supports native shell/apply_patch tools."
        )

    # Set base path - use current directory by default
    if work_dir:
        base_path = os.path.abspath(work_dir)
    else:
        base_path = os.getcwd()

    if not os.path.exists(base_path):
        raise ValueError(f"Directory not found: {base_path}")

    print(f"📁 Working directory: {base_path}")

    # Require HUD_API_KEY for gateway access
    if not settings.api_key:
        raise ValueError(
            "HUD_API_KEY is required.\n"
            "Get yours at: https://hud.ai/project/api-keys\n"
            "Then: export HUD_API_KEY='sk-hud-...'"
        )

    # Create environment with Codex tools - 1:1 match with OpenAI's Codex CLI
    # Both tools use the same working directory for consistency
    env = hud.Environment("local-codex")
    env.add_tool(ShellTool(cwd=base_path))
    env.add_tool(ApplyPatchTool(base_path=base_path))

    # Create agent using HUD Gateway (uses HUD_API_KEY)
    model_client = AsyncOpenAI(
        base_url=settings.hud_gateway_url,
        api_key=settings.api_key,
    )
    agent = OpenAIAgent.create(
        model=model,
        model_client=model_client,
        validate_api_key=False,  # HUD key won't validate against OpenAI
        verbose=verbose,
    )
    print("🌐 Using HUD Gateway for inference")

    print(f"🤖 Model: {model}")
    print(f"📋 Task: {task}")
    print("=" * 60)

    # Define a scenario for the coding task
    @env.scenario("coding_task")
    async def coding_task_scenario(task_description: str):
        yield f"""You are a skilled software developer. Complete the following task:

{task_description}

Use the available tools:
- `shell` to run commands (ls, cat, python, etc.)
- `apply_patch` to create or modify files

Work in the current directory. When done, verify your work runs correctly."""

        # Simple success - task completed
        yield 1.0

    # Run the agent
    eval_task = env("coding_task", task_description=task)

    async with hud.eval(eval_task, name="codex-coding-local") as ctx:
        await agent.run(ctx, max_steps=max_steps)

    print("=" * 60)
    print("✅ Task completed!")
    print(f"📊 Reward: {ctx.reward}")


# =============================================================================
# Run Coding Task via HUD Hub
# =============================================================================


async def run_coding_task_hub(
    task: str,
    model: str = "gpt-5.1",
    max_steps: int = 20,
    hub_name: str = DEFAULT_HUB,
    verbose: bool = False,
) -> None:
    """
    Run a coding task against the codex_environment_sandbox via HUD Hub.

    Uses connect_hub() to route through HUD's infrastructure, enabling
    full telemetry (both inference and environment steps visible in trace).

    Note: You must create the codex_environment_sandbox environment in hud.ai
    first before using this function.

    Args:
        task: Description of the coding task
        model: OpenAI model to use (default: gpt-5.1)
        max_steps: Maximum agent steps (default: 20)
        hub_name: Hub environment name (default: codex_environment_sandbox)
        verbose: Enable verbose output
    """
    # Require HUD_API_KEY for gateway access
    if not settings.api_key:
        raise ValueError(
            "HUD_API_KEY is required.\n"
            "Get yours at: https://hud.ai/project/api-keys\n"
            "Then: export HUD_API_KEY='sk-hud-...'"
        )

    print(f"🌐 Connecting to hub: {hub_name}")

    # Create environment and connect via HUD Hub (full telemetry)
    env = hud.Environment()
    env.connect_hub(hub_name)

    # Validate model is Codex-capable
    if model not in CODEX_MODELS:
        raise ValueError(
            f"Model '{model}' is not in the Codex-capable list {sorted(CODEX_MODELS)}.\n"
            "Use a model that supports native shell/apply_patch tools."
        )

    # Create agent with HUD Gateway for inference telemetry
    model_client = AsyncOpenAI(
        base_url=settings.hud_gateway_url,
        api_key=settings.api_key,
    )
    agent = OpenAIAgent.create(
        model=model,
        model_client=model_client,
        validate_api_key=False,  # HUD key won't validate against OpenAI
        verbose=verbose,
    )
    print("🌐 Using HUD Gateway for inference")

    print(f"🤖 Model: {model}")
    print(f"📋 Task: {task}")
    print("=" * 60)

    # Define a scenario for the coding task
    @env.scenario("coding_task")
    async def coding_task_scenario(task_description: str):
        yield f"""You are a skilled software developer. Complete the following task:

{task_description}

Use the available tools:
- `shell` to run commands (ls, cat, python, etc.)
- `apply_patch` to create or modify files

Work in the current directory. When done, verify your work runs correctly."""

        # Evaluation is handled by the environment's evaluate tool
        yield 1.0

    # Run the agent
    eval_task = env("coding_task", task_description=task)

    async with hud.eval(eval_task, name="codex-coding") as ctx:
        await agent.run(ctx, max_steps=max_steps)

    print("=" * 60)
    print("✅ Task completed!")
    print(f"📊 Reward: {ctx.reward}")


# =============================================================================
# CLI
# =============================================================================


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run coding tasks with OpenAI's native shell and apply_patch tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local mode (no Docker, no HUD_API_KEY required)
  uv run python examples/06_codex_coding_agent.py --local

  # Local mode with custom working directory
  uv run python examples/06_codex_coding_agent.py --local --work-dir ./codex_output

  # Hub mode (full telemetry, requires HUD_API_KEY)
  uv run python examples/06_codex_coding_agent.py

  # Custom task
  uv run python examples/06_codex_coding_agent.py --local \\
    --task "Create a Python script that prints the Fibonacci sequence up to 10 numbers"

  # Verbose output
  uv run python examples/06_codex_coding_agent.py --local --verbose

  # Use a different Codex model
  uv run python examples/06_codex_coding_agent.py --local --model gpt-5.1-codex
""",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run locally without Docker (tools execute on your machine)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Create a Python script called main.py that prints 'Hello, World!' and the current date/time",
        help="The coding task to complete",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.1",
        help="Codex-capable OpenAI model (default: gpt-5.1)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Maximum agent steps (default: 20)",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Working directory for file operations (default: current directory)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()

    if args.local:
        await run_coding_task_local(
            task=args.task,
            model=args.model,
            max_steps=args.max_steps,
            verbose=args.verbose,
            work_dir=args.work_dir,
        )
    else:
        await run_coding_task_hub(
            task=args.task,
            model=args.model,
            max_steps=args.max_steps,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    asyncio.run(main())
