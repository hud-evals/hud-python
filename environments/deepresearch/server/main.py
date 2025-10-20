"""Controller bridges MCP tools to the DeepResearch environment HTTP API."""

from typing import List, Dict
import httpx
import os
import sys
import logging

from hud.tools.types import EvaluationResult

from hud.server import MCPServer

# Configure logging
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
    force=True,  # Force all loggers to use stderr
)

# MCP server
mcp = MCPServer(name="deepresearch")

# Environment server URL (backend)
ENV_SERVER_URL = os.getenv("ENV_SERVER_URL", "http://localhost:8000")

# Shared HTTP client to talk to the environment
http_client = httpx.AsyncClient(
    base_url=ENV_SERVER_URL,
    timeout=30.0,
    headers={"User-Agent": "HUD-DeepResearch-Controller/1.0"},
)


@mcp.initialize
async def init():
    # Ensure environment server is reachable
    await http_client.get("/health")


@mcp.shutdown
async def cleanup():
    await http_client.aclose()


@mcp.tool()
async def setup() -> str:
    await http_client.post("/setup")
    return "Environment setup"


@mcp.tool()
async def search(query: str) -> List[Dict[str, str]]:
    resp = await http_client.post("/search", json={"query": query})
    return resp.json()


@mcp.tool()
async def fetch(url: str) -> str:
    resp = await http_client.post("/fetch", json={"url": url})
    data = resp.json()
    return data.get("content", "")


@mcp.tool()
async def answer(final_answer: str) -> str:
    await http_client.post("/answer", json={"final_answer": final_answer})
    return f"Answer submitted: {final_answer}"


@mcp.tool()
async def evaluate(expected_answer: str) -> EvaluationResult:
    resp = await http_client.post("/evaluate", json={"expected_answer": expected_answer})
    return EvaluationResult(**resp.json())


if __name__ == "__main__":
    mcp.run()
