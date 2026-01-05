#!/usr/bin/env python3
"""
Codex Coding Agent Example

This example demonstrates how to use OpenAI's **Codex-capable** models with
native `shell` and `apply_patch` tools via the HUD SDK.

What this shows:
- **Local mode**: Run locally without Docker - tools execute on your machine
- **Hub mode**: Connect to HUD Hub for full telemetry and cloud execution
- OpenAIAgent automatically converts tools to OpenAI's native tool types

Usage:
  # Local mode (no Docker required, no HUD_API_KEY required for OPENAI_API_KEY users)
  uv run python examples/06_codex_coding_agent.py --local

  # Hub mode (requires HUD_API_KEY)
  export HUD_API_KEY="sk-hud-..."
  uv run python examples/06_codex_coding_agent.py

  # Custom task
  uv run python examples/06_codex_coding_agent.py --local \\
    --task "Create a Python script that prints the Fibonacci sequence"

Requirements:
  - Install deps: `uv sync`
  - For local mode: OPENAI_API_KEY environment variable
  - For hub mode: HUD_API_KEY environment variable
  - For traces (hud.eval): HUD_API_KEY environment variable
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
from hud.tools.apply_patch import ApplyPatchTool
from hud.tools.shell import ShellTool

# =============================================================================
# Configuration
# =============================================================================

# Default hub environment name
DEFAULT_HUB = "codex_sandbox_environment"

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

    # Create working directory
    if work_dir:
        os.makedirs(work_dir, exist_ok=True)
        base_path = os.path.abspath(work_dir)
    else:
        # Default to ./codex_output
        work_dir = "./codex_output"
        os.makedirs(work_dir, exist_ok=True)
        base_path = os.path.abspath(work_dir)

    print(f"📁 Working directory: {base_path}")

    # Initialize tools
    shell_tool = ShellTool()
    apply_patch_tool = ApplyPatchTool(base_path=base_path)

    # Create environment with local tools
    env = hud.Environment("local-codex")

    @env.tool()
    async def shell(
        commands: list[str],
        timeout_ms: int | None = None,
        max_output_length: int | None = None,
    ) -> dict:
        """Execute shell commands in a bash session.

        Args:
            commands: List of shell commands to execute
            timeout_ms: Optional timeout in milliseconds for each command
            max_output_length: Optional max output length hint
        """
        # Change to working directory before executing
        prefixed_commands = [f"cd {base_path} && {cmd}" for cmd in commands]
        result = await shell_tool(
            commands=prefixed_commands,
            timeout_ms=timeout_ms,
            max_output_length=max_output_length,
        )
        return result.to_dict()

    @env.tool()
    async def apply_patch(
        type: str,
        path: str,
        diff: str | None = None,
    ) -> dict:
        """Apply file operations using V4A diff format.

        Args:
            type: Operation type - "create_file", "update_file", or "delete_file"
            path: The file path to operate on
            diff: The diff content (required for create_file and update_file)
        """
        result = await apply_patch_tool(type=type, path=path, diff=diff)
        return result.to_dict()

    # Create OpenAI client
    model_client = AsyncOpenAI()
    agent = OpenAIAgent.create(
        model=model,
        model_client=model_client,
        verbose=verbose,
    )

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
    print(f"📁 Files created in: {base_path}")

    # List created files
    if os.path.exists(base_path):
        files = os.listdir(base_path)
        if files:
            print(f"📄 Files: {', '.join(files)}")


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
    Run a coding task against the codex-sandbox environment via HUD Hub.

    Uses connect_hub() to route through HUD's infrastructure, enabling
    full telemetry (both inference and environment steps visible in trace).

    Args:
        task: Description of the coding task
        model: OpenAI model to use (default: gpt-5.1)
        max_steps: Maximum agent steps (default: 20)
        hub_name: Hub environment name (default: codex-sandbox)
        verbose: Enable verbose output
    """
    # Require HUD_API_KEY for hub mode
    if not settings.api_key:
        raise ValueError(
            "HUD_API_KEY is required for hub mode.\n"
            "Get yours at: https://hud.ai/project/api-keys\n"
            "Then: export HUD_API_KEY='sk-hud-...'\n\n"
            "Or use --local flag to run without HUD infrastructure."
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
        help="Working directory for file operations (local mode only, default: ./codex_output)",
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
