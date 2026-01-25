#!/usr/bin/env python3
"""
Build Your Own OpenCode - A Recreation of the OpenCode Coding Agent

This example shows how to build your own OpenCode (https://github.com/anomalyco/opencode)
from scratch using the HUD SDK. OpenCode is a popular open-source coding agent that uses:

- `str_replace` editing via EditTool (same as OpenCode's edit tool)
- Filesystem exploration via ReadTool, GrepTool, GlobTool, ListTool
- Shell execution via ShellTool

What you get:
- **Your own OpenCode** - Same tools as OpenCode, fully customizable
- **Full observability** - Every tool call and response traced on hud.ai
- **Plan mode** - Read-only agent for safe codebase exploration

Usage:
  # Build mode - full coding capabilities
  uv run python examples/07_opencode_agent.py --task "Fix the bug in main.py"

  # Plan mode - read-only exploration
  uv run python examples/07_opencode_agent.py --plan --task "How does auth work?"

  # Verbose output
  uv run python examples/07_opencode_agent.py --verbose --task "Add error handling"

Requirements:
  - Install deps: `uv sync`
  - Set HUD_API_KEY environment variable (get at hud.ai/project/api-keys)
"""

import argparse
import asyncio
import os

from dotenv import load_dotenv

# Load .env file from current directory or parent directories
load_dotenv()

import hud
from hud.agents import create_agent
from hud.tools.coding import EditTool, ShellTool
from hud.tools.filesystem import GlobTool, GrepTool, ListTool, ReadTool


# =============================================================================
# Run Coding Task (Build Mode)
# =============================================================================


async def run_build_mode(
    task: str,
    model: str = "gpt-4o",
    max_steps: int = 30,
    verbose: bool = False,
    work_dir: str | None = None,
) -> None:
    """
    Run a coding task with full build capabilities.

    Uses ShellTool, EditTool, and filesystem tools for complete
    coding agent functionality.

    Args:
        task: Description of the coding task
        model: Model to use (default: gpt-4o)
        max_steps: Maximum agent steps (default: 30)
        verbose: Enable verbose output
        work_dir: Working directory for file operations
    """
    # Create working directory
    if work_dir:
        os.makedirs(work_dir, exist_ok=True)
        base_path = os.path.abspath(work_dir)
    else:
        work_dir = "./opencode_output"
        os.makedirs(work_dir, exist_ok=True)
        base_path = os.path.abspath(work_dir)

    print(f"📁 Working directory: {base_path}")

    # Create environment with OpenCode tools
    env = hud.Environment("opencode-build")

    # Coding tools
    env.add_tool(ShellTool())
    env.add_tool(EditTool())

    # Filesystem exploration tools
    env.add_tool(ReadTool(base_path=base_path))
    env.add_tool(GrepTool(base_path=base_path))
    env.add_tool(GlobTool(base_path=base_path))
    env.add_tool(ListTool(base_path=base_path))

    # Create agent
    agent = create_agent(model, verbose=verbose)

    print(f"🤖 Model: {model}")
    print(f"📋 Task: {task}")
    print("=" * 60)

    # Define scenario for evaluation
    @env.scenario("coding_task")
    async def coding_task_scenario(task_description: str):
        yield f"""You are a skilled software developer. Complete the following task:

{task_description}

Available tools:
- `shell` - Run bash commands
- `str_replace_based_edit_tool` - Edit files using str_replace
- `read` - Read file contents with line numbers
- `grep` - Search file contents with regex
- `glob` - Find files by pattern
- `list` - List directory contents

Explore the codebase first using read/grep/glob/list, then make changes."""
        yield 1.0

    # Run the agent
    eval_task = env("coding_task", task_description=task)

    async with hud.eval(eval_task, name="opencode-build") as ctx:
        await agent.run(ctx, max_steps=max_steps)

    print("=" * 60)
    print("✅ Task completed!")
    print(f"📊 Reward: {ctx.reward}")
    print(f"📁 Files in workspace: {base_path}")

    # List created files
    if os.path.exists(base_path):
        files = os.listdir(base_path)
        if files:
            print(f"📄 Files: {', '.join(files)}")


# =============================================================================
# Run Plan Mode (Read-Only)
# =============================================================================


async def run_plan_mode(
    question: str,
    model: str = "gpt-4o",
    max_steps: int = 20,
    verbose: bool = False,
    work_dir: str | None = None,
) -> None:
    """
    Run in plan mode - read-only codebase exploration.

    Only uses filesystem exploration tools (no edit or shell).
    Safe for analyzing codebases without making changes.

    Args:
        question: Question to answer about the codebase
        model: Model to use (default: gpt-4o)
        max_steps: Maximum agent steps (default: 20)
        verbose: Enable verbose output
        work_dir: Directory to explore
    """
    # Set base path
    if work_dir:
        base_path = os.path.abspath(work_dir)
    else:
        base_path = os.getcwd()

    if not os.path.exists(base_path):
        raise ValueError(f"Directory not found: {base_path}")

    print(f"📁 Exploring: {base_path}")

    # Create environment with read-only tools
    env = hud.Environment("opencode-plan")

    # Only filesystem exploration - no coding tools
    env.add_tool(ReadTool(base_path=base_path))
    env.add_tool(GrepTool(base_path=base_path))
    env.add_tool(GlobTool(base_path=base_path))
    env.add_tool(ListTool(base_path=base_path))

    # Create agent
    agent = create_agent(model, verbose=verbose)

    print(f"🤖 Model: {model}")
    print(f"❓ Question: {question}")
    print("=" * 60)

    # Define scenario
    @env.scenario("analyze")
    async def analyze_scenario(query: str):
        yield f"""You are analyzing a codebase. Answer this question:

{query}

Available tools:
- `read` - Read file contents with line numbers
- `grep` - Search file contents with regex
- `glob` - Find files by pattern
- `list` - List directory contents

Use these read-only tools to explore. Do NOT suggest code changes."""
        yield 1.0

    # Run the agent
    eval_task = env("analyze", query=question)

    async with hud.eval(eval_task, name="opencode-plan") as ctx:
        await agent.run(ctx, max_steps=max_steps)

    print("=" * 60)
    print("✅ Analysis complete!")
    print(f"📊 Reward: {ctx.reward}")


# =============================================================================
# CLI
# =============================================================================


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenCode-style coding agent with HUD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build mode - make changes to code
  uv run python examples/07_opencode_agent.py --task "Add error handling to api.py"

  # Plan mode - read-only exploration
  uv run python examples/07_opencode_agent.py --plan --task "How does auth work?"

  # Custom working directory
  uv run python examples/07_opencode_agent.py --work-dir ./my-project --task "Fix bugs"

  # Verbose output
  uv run python examples/07_opencode_agent.py --verbose --task "Refactor utils"

  # Use Claude
  uv run python examples/07_opencode_agent.py --model claude-sonnet-4-5 --task "Add tests"
""",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Run in plan mode (read-only, no edits)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Create a Python script that prints Hello World",
        help="The task to complete (build mode) or question to answer (plan mode)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model to use (default: gpt-4o)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Maximum agent steps (default: 30)",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Working directory (default: ./opencode_output for build, cwd for plan)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()

    if args.plan:
        await run_plan_mode(
            question=args.task,
            model=args.model,
            max_steps=args.max_steps,
            verbose=args.verbose,
            work_dir=args.work_dir,
        )
    else:
        await run_build_mode(
            task=args.task,
            model=args.model,
            max_steps=args.max_steps,
            verbose=args.verbose,
            work_dir=args.work_dir,
        )


if __name__ == "__main__":
    asyncio.run(main())
