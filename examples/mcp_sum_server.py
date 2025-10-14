"""FastMCP server exposing a simple sum tool.

Run with: `python examples/mcp_sum_server.py`.
"""

from __future__ import annotations

from fastmcp import FastMCP


server = FastMCP("SumServer")


@server.tool()
def sum(a: int, b: int) -> dict[str, int]:
    """Return the sum of two integers."""
    return {"result": a + b}


if __name__ == "__main__":
    server.run()


