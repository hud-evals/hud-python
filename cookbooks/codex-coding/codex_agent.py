#!/usr/bin/env python3
"""
Build Your Own Codex - A Recreation of OpenAI's Codex CLI

This cookbook shows how to build your own Codex (https://github.com/openai/codex)
from scratch using the HUD SDK. The environment runs a ``Workspace`` serving an
``ssh`` capability; the ``OpenAIAgent`` drives it with OpenAI's native
``shell`` and ``apply_patch`` tools — the same protocol the ``codex`` CLI uses.

What you get:
- **Your own Codex** - Same behavior as `codex` CLI, but fully customizable
- **Full observability** - Every tool call and response traced on hud.ai

See the README in this directory for setup and usage. Requires ``HUD_API_KEY``
(gateway inference).
"""

import argparse
import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load .env file from current directory or parent directories
load_dotenv()

import hud
from hud import LocalRuntime
from hud.agents.openai import OpenAIAgent
from hud.agents.types import OpenAIConfig
from hud.settings import settings

# Codex-capable models that support native shell/apply_patch tools
CODEX_MODELS = {
    "gpt-5.1-codex",
    "gpt-5.1",
    "gpt-5.3-codex",
    "gpt-5.4",
    "gpt-5.6",
}

PROMPT_TEMPLATE = """You are a skilled software developer. Complete the following task:

{task_description}

Use the available tools:
- `shell` to run commands (ls, cat, python, etc.)
- `apply_patch` to create or modify files

Work in the current directory. When done, verify your work runs correctly."""

# The environment this file *is*: `LocalRuntime(__file__)` serves it in a child
# process (which re-imports this module), so the task's prompt and grade
# arrive over the wire while the agent loop runs here. The workspace root is
# handed to that child via CODEX_WORK_DIR. Attaching the workspace writes
# nothing: the serving child starts it (SSH keys + socket) and publishes the
# shell capability when the env comes up.
WORK_DIR = os.path.abspath(os.environ.get("CODEX_WORK_DIR") or os.getcwd())
env = hud.Environment("local-codex")
env.workspace(WORK_DIR)


@env.template()
async def coding_task(task_description: str):
    yield PROMPT_TEMPLATE.format(task_description=task_description)
    yield 1.0  # simple success - task completed


async def run_coding_task(
    task: str,
    model: str = "gpt-5.3-codex",
    max_steps: int = 20,
    work_dir: str | None = None,
) -> None:
    """Run a coding task locally.

    The environment runs a ``Workspace`` on your machine serving an ``ssh``
    capability; the agent's shell commands and patches land in that directory.
    """
    if model not in CODEX_MODELS:
        raise ValueError(
            f"Model '{model}' is not in the Codex-capable list {sorted(CODEX_MODELS)}.\n"
            "Use a model that supports native shell/apply_patch tools."
        )
    if not settings.api_key:
        raise ValueError(
            "HUD_API_KEY is required.\n"
            "Get yours at: https://hud.ai/project/api-keys\n"
            "Then: export HUD_API_KEY='sk-hud-...'"
        )

    base_path = os.path.abspath(work_dir) if work_dir else os.getcwd()
    if not os.path.exists(base_path):
        raise ValueError(f"Directory not found: {base_path}")
    os.environ["CODEX_WORK_DIR"] = base_path  # inherited by the spawned env process

    print(f"📁 Working directory: {base_path}")

    # Codex-capable OpenAIAgent routed through the HUD gateway.
    model_client = AsyncOpenAI(
        base_url=settings.hud_gateway_url,
        api_key=settings.api_key,
    )
    agent = OpenAIAgent(OpenAIConfig(model=model, model_client=model_client, max_steps=max_steps))

    print("🌐 Using HUD Gateway for inference")
    print(f"🤖 Model: {model}")
    print(f"📋 Task: {task}")
    print("=" * 60)

    job = await coding_task(task_description=task).run(agent, runtime=LocalRuntime(__file__))

    print("=" * 60)
    (run,) = job.runs
    if run.trace.is_error:
        print(f"❌ Task failed: {run.trace.content}")
        return
    print("✅ Task completed!")
    print(f"📊 Reward: {job.reward}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run coding tasks with OpenAI's native shell and apply_patch tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run codex_agent.py

  # Custom working directory
  uv run codex_agent.py --work-dir ./codex_output

  # Custom task
  uv run codex_agent.py \\
    --task "Create a Python script that prints the Fibonacci sequence up to 10 numbers"

  # Use a different Codex model
  uv run codex_agent.py --model gpt-5.1-codex
""",
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
        default="gpt-5.3-codex",
        help="Codex-capable OpenAI model (default: gpt-5.3-codex)",
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
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    await run_coding_task(
        task=args.task,
        model=args.model,
        max_steps=args.max_steps,
        work_dir=args.work_dir,
    )


if __name__ == "__main__":
    asyncio.run(main())
