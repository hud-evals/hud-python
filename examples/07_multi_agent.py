"""
Multi-Agent Example - Smart Research Assistant

This example demonstrates how to compose multiple specialized agents
into a multi-agent system using AgentTools.

The pattern is simple:
1. Create AgentTools that wrap environments + models
2. Register them on a coordinator environment
3. Run a "conductor" agent that dispatches work to sub-agents

The Smart Research Assistant combines:
- Browser agent: Finds information, scrapes data, navigates websites
- Coding agent: Creates markdown files with research findings

Uses real HUD Hub environments:
- codex_environment_sandbox: Coding environment with shell and file editing tools
- hud-remote-browser-2: Browser automation for web tasks

Usage:
    export HUD_API_KEY="sk-hud-..."
    uv run python examples/07_multi_agent.py

    # Custom task
    uv run python examples/07_multi_agent.py \\
        --task "Find current prices of Bitcoin and Ethereum and save to crypto.md"
"""

import argparse
import asyncio

from dotenv import load_dotenv

load_dotenv()

import hud
from hud import Environment
from hud.agents import create_agent
from hud.settings import settings
from hud.tools.agent import AgentTool


# =============================================================================
# Create Sub-Agents from Hub Environments
# =============================================================================


def create_coding_agent() -> AgentTool:
    """Create a coding sub-agent for markdown file creation."""
    env = Environment("coding")
    env.connect_hub("codex_environment_sandbox")

    @env.scenario()
    async def create_markdown(
        filename: str,
        content: str,
        expected_result: str | None = None,  # Eval-only param (hidden from tool schema)
    ):
        """Create a markdown file with the given content."""
        prompt = f"""You are a file creation assistant with access to a coding environment.

Task: Create a markdown file named '{filename}' with the following content:

{content}

IMPORTANT: Use the `apply_patch` tool to create the file. Do NOT use shell commands like cat or echo.

Steps:
1. Use apply_patch to create '{filename}' with the content above
2. Use list_files or read_file to confirm it was created

Return a confirmation message with the filename and location."""

        yield prompt
        yield 1.0

    return AgentTool(
        env("create_markdown"),
        model="gpt-5.1",
        name="create_markdown",
        description="Create a markdown file with specified content. Use for: "
        "saving research findings, creating reports, documenting results.",
    )


def create_browser_agent() -> AgentTool:
    """Create a browser automation sub-agent for web research."""
    env = Environment("browser")
    env.connect_hub("hud-remote-browser-2")

    @env.scenario()
    async def web_research(
        task: str,
        start_url: str | None = None,
        expected_outcome: str | None = None,  # Eval-only param
    ):
        """Research information on the web using browser automation."""
        prompt = f"""You are a web research agent with access to a browser.

Research Task: {task}
"""
        if start_url:
            prompt += f"\nStart URL: {start_url}"

        prompt += """

Your job is to:
1. Navigate to relevant websites
2. Search for information related to the task
3. Extract key data, facts, and information
4. Return a clear, structured summary

Include: key findings, data points (prices, numbers, dates), and sources visited."""

        yield prompt
        yield 1.0

    return AgentTool(
        env("web_research"),
        model="claude-sonnet-4-5",
        name="web_research",
        description="Research information on the web. Use for finding articles, "
        "scraping data, comparing prices, and extracting structured information.",
    )


# =============================================================================
# Multi-Agent Orchestration Pattern
# =============================================================================


async def run_multi_agent(
    task: str,
    conductor_model: str = "gpt-4o",
    max_steps: int = 10,
    verbose: bool = False,
) -> None:
    """
    Run a multi-agent system with a conductor dispatching to sub-agents.

    This shows the core pattern for multi-agent orchestration:
    1. Create an Environment for the coordinator
    2. Add AgentTools as callable tools
    3. Run a conductor agent that dispatches work
    """

    if not settings.api_key:
        raise ValueError(
            "HUD_API_KEY is required for hub environments.\n"
            "Get yours at: https://hud.ai/project/api-keys\n"
            "Then: export HUD_API_KEY='sk-hud-...'"
        )

    # Create sub-agents as tools
    coding_agent = create_coding_agent()
    browser_agent = create_browser_agent()

    # Create coordinator environment with sub-agents as tools
    coordinator = Environment("coordinator")
    coordinator.add_tool(browser_agent)
    coordinator.add_tool(coding_agent)

    # Define the coordination scenario
    @coordinator.scenario()
    async def coordinate(prompt: str):
        yield prompt
        yield 1.0

    # System prompt for the conductor
    system_prompt = """You are a Smart Research Assistant coordinating specialized agents.

Available sub-agents (call as tools):
- web_research: Find information, scrape data, compare prices
- create_markdown: Create markdown files with specified content

CRITICAL: Sub-agents don't share context. When calling create_markdown,
you MUST pass the full content you want to save.

Workflow:
1. Use web_research to gather data (prices, facts, numbers)
2. Format the data into markdown content
3. Use create_markdown to save the formatted content
4. Iterate if needed"""

    print("🎭 Smart Research Assistant")
    print(f"🤖 Conductor: {conductor_model}")
    print(f"🔧 Sub-agents: {browser_agent.name}, {coding_agent.name}")
    print(f"📋 Task: {task}")
    print("=" * 70)

    # Run with eval context
    async with hud.eval(
        coordinator("coordinate", prompt=task),
        name="multi-agent-research",
    ) as ctx:
        # Create conductor agent and run
        conductor = create_agent(
            conductor_model,
            system_prompt=system_prompt,
            verbose=verbose,
        )
        result = await conductor.run(ctx, max_steps=max_steps)

    print("=" * 70)
    print("✅ Research Complete!")
    print(f"📊 Reward: {ctx.reward}")
    if result.content:
        print(f"\n📝 Summary:\n{result.content}")


# =============================================================================
# CLI
# =============================================================================


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-agent research assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python examples/07_multi_agent.py \\
    --task "Research AAPL stock price and save to stock_prices.md"

  uv run python examples/07_multi_agent.py \\
    --task "Find 3 laptops under $2000 and save specs to laptops.md"
""",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Research current price of GOOGL and save to googl_price.md",
        help="Research task to complete",
    )
    parser.add_argument(
        "--conductor",
        type=str,
        default="gpt-4o",
        help="Model for the conductor agent (default: gpt-4o)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum steps for conductor (default: 10)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()

    await run_multi_agent(
        task=args.task,
        conductor_model=args.conductor,
        max_steps=args.max_steps,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    asyncio.run(main())
