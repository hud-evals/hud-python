"""Using the Claude Agent SDK with HUD environments.

Instead of implementing an agent loop manually (like ClaudeAgent does),
you can use the Claude Agent SDK which manages the full loop for you.
Just hand it your environment's tools and a prompt.

┌───────────────────┐  MCP tools   ┌───────────────┐
│  Claude Agent SDK │ ◄──────────► │  Environment  │
│  (manages loop)   │              │  (hud.Env)    │
└───────────────────┘              └───────────────┘

API routing:
  - If HUD_API_KEY is set, Claude API calls are automatically routed
    through the HUD inference gateway (for billing/telemetry).
  - Otherwise falls back to ANTHROPIC_API_KEY directly.

Requires: pip install claude-agent-sdk

Three patterns shown below:
  1. Low-level: get MCP server, configure SDK yourself
  2. Convenience: as_claude_agent_options() does it all (inc. gateway)
  3. Local env: no remote hub needed
"""

from __future__ import annotations

import asyncio

import hud
from hud.eval.task import Task

# ------------------------------------------------------------------
# Pattern 1: Low-level — bring your own SDK config
# ------------------------------------------------------------------


async def low_level() -> None:
    """Use as_claude_agent_mcp_server() for full control over SDK config."""
    from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, query

    from hud.settings import settings

    task = Task(
        env={"name": "browser"},
        scenario="navigate",
        args={"url": "https://example.com"},
    )

    async with hud.eval(task, name="sdk-low-level") as ctx:
        # Get an in-process MCP server wrapping the environment's tools
        server = ctx.as_claude_agent_mcp_server(server_name="env")

        # Build tool allowlist (SDK naming: mcp__<server>__<tool>)
        tool_names = [f"mcp__env__{t.name}" for t in ctx.as_tools()]

        # Route through HUD gateway if HUD_API_KEY is set
        sdk_env = {}
        if settings.api_key:
            sdk_env = {
                "ANTHROPIC_API_KEY": settings.api_key,
                "ANTHROPIC_BASE_URL": settings.hud_gateway_url,
            }

        options = ClaudeAgentOptions(
            mcp_servers={"env": server},
            allowed_tools=tool_names,
            permission_mode="bypassPermissions",
            max_turns=15,
            env=sdk_env,
        )

        # Run the SDK — it handles the entire agent loop
        final_text = ""
        async for message in query(prompt=ctx.prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        final_text = block.text

        # Submit answer for scenario evaluation
        if final_text and ctx.has_scenario:
            await ctx.submit(final_text)

    print(f"Reward: {ctx.reward}")


# ------------------------------------------------------------------
# Pattern 2: Convenience — one-liner (handles gateway automatically)
# ------------------------------------------------------------------


async def convenience() -> None:
    """Use as_claude_agent_options() — gateway routing is automatic."""
    from claude_agent_sdk import AssistantMessage, TextBlock, query

    task = Task(
        env={"name": "browser"},
        scenario="navigate",
        args={"url": "https://example.com"},
    )

    async with hud.eval(task, name="sdk-convenience") as ctx:
        # One call builds: MCP server + tool allowlist + gateway env
        # If HUD_API_KEY is set, API calls route through HUD gateway
        options = ctx.as_claude_agent_options(
            model="sonnet",
            max_turns=15,
        )

        final_text = ""
        async for message in query(prompt=ctx.prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        final_text = block.text

        if final_text and ctx.has_scenario:
            await ctx.submit(final_text)

    print(f"Reward: {ctx.reward}")


# ------------------------------------------------------------------
# Pattern 3: Local env (no remote hub needed)
# ------------------------------------------------------------------

env = hud.Environment("calc")


@env.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@env.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


async def local_env() -> None:
    """Use Claude Agent SDK with a local environment."""
    from claude_agent_sdk import AssistantMessage, TextBlock, query

    async with hud.eval(env(), trace=False) as ctx:
        # Gateway routing still applies here if HUD_API_KEY is set
        options = ctx.as_claude_agent_options()

        async for message in query(
            prompt="What is (3 + 4) * 5? Use the tools to compute it step by step.",
            options=options,
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(block.text)


if __name__ == "__main__":
    asyncio.run(local_env())
