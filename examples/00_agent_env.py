"""Tiny agent-environment demo in one file.

┌───────────────┐  tool call (MCP)  ┌───────────────┐
│   Client      │ ────────────────► │  Server       │
│ (agent side)  │  JSON-RPC / stdio │ (environment) │
└───────────────┘                   └───────────────┘

Server = the *environment*
• Exposes one tool `sum(a, b)` using the FastMCP SDK.
• In real projects the server runs inside Docker so stdout is reserved for the
  protocol and stderr for logs.

Client = the *agent side*
• Uses `hud.client.MCPClient` to connect to **any** MCP environment – local
  subprocess here, Docker or remote HUD in real scenarios.
• Sends a single tool call and prints the result.

Run `python examples/00_minimal_fastmcp.py` → prints `3 + 4 = 7`.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from fastmcp import FastMCP
from hud.client import MCPClient

# ------------------------------------------------------------------
# Environment (server)
# ------------------------------------------------------------------

server = FastMCP("MiniServer")


@server.tool()
def sum(a: int, b: int) -> int:
    return a + b


# ------------------------------------------------------------------
# Agent (client) – spawns the same file with --server and calls the tool
# ------------------------------------------------------------------

THIS_FILE = Path(__file__).absolute()


async def run_client() -> None:
    cfg = {
        "local": {
            "command": sys.executable,
            "args": [str(THIS_FILE), "--server"],
        }
    }
    client = MCPClient(mcp_config=cfg)
    await client.initialize()
    result = await client.call_tool("sum", {"a": 3, "b": 4})
    print("3 + 4 =", result)
    await client.close()


if __name__ == "__main__":
    if "--server" in sys.argv:
        server.run()
    else:
        asyncio.run(run_client())  # The client will run itself with the --server flag
