"""Tiny agent-environment demo in one file.

┌───────────────┐  tool call (MCP)  ┌───────────────┐
│   Agent       │ ────────────────► │  Environment  │
│ (client)      │   hud.eval()      │  (hud.Env)    │
└───────────────┘                   └───────────────┘

Environment = hud.Environment with @env.tool
• Exposes one tool `sum(a, b)` using the @env.tool decorator.
• In real projects this would be a Docker image or remote service.

Agent = the client side
• Uses `hud.eval(env())` to connect and call tools.
• The environment handles tool routing automatically.

Run `python examples/00_agent_env.py` → prints `3 + 4 = 7`.
"""

from __future__ import annotations

import asyncio

import hud

# ------------------------------------------------------------------
# Environment (with local tools)
# ------------------------------------------------------------------

env = hud.Environment("calculator")


@env.tool()
def sum(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


# ------------------------------------------------------------------
# Agent (client) – connects to env and calls tools
# ------------------------------------------------------------------


async def main() -> None:
    """Connect to the environment and call the sum tool."""
    # Use hud.eval() with env() to create a task and run it
    async with hud.eval(env(), trace=False) as ctx:
        result = await ctx.call_tool(name="sum", arguments={"a": 3, "b": 4})
        print("3 + 4 =", result)


if __name__ == "__main__":
    asyncio.run(main())
