"""MCP server for SEC EDGAR research environment."""

from typing import List, Dict, Any, Optional
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
mcp = MCPServer(name="sec-rubrics")

# Environment server URL (backend)
ENV_SERVER_URL = os.getenv("ENV_SERVER_URL", "http://localhost:8000")

# Shared HTTP client to talk to the environment
http_client = httpx.AsyncClient(
    base_url=ENV_SERVER_URL,
    timeout=60.0,  # Increased timeout for SEC EDGAR operations
    headers={"User-Agent": "HUD-SEC-Rubrics-Controller/1.0"},
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
    return "Environment setup complete"


@mcp.tool()
async def search_company(query: str) -> List[Dict[str, str]]:
    resp = await http_client.post("/search_company", json={"query": query})
    return resp.json()


@mcp.tool()
async def get_filings(
    ticker: str, form_type: Optional[str] = None, limit: int = 10
) -> List[Dict[str, Any]]:
    resp = await http_client.post(
        "/get_filings", json={"ticker": ticker, "form_type": form_type, "limit": limit}
    )
    return resp.json()


@mcp.tool()
async def get_filing_content(filing_url: str) -> str:
    resp = await http_client.post("/get_filing_content", json={"filing_url": filing_url})
    data = resp.json()
    return data.get("content", "")


@mcp.tool()
async def get_recent_filings(
    identifier: Optional[str] = None, form_type: Optional[str] = None, limit: int = 50
) -> Any:
    resp = await http_client.post(
        "/get_recent_filings",
        json={
            "identifier": identifier,
            "form_type": form_type,
            "limit": limit,
        },
    )
    return resp.json()


@mcp.tool()
async def get_filing_content_by_accession(identifier: str, accession_number: str) -> Any:
    """Get filing content using identifier (ticker/CIK) and accession number."""
    resp = await http_client.post(
        "/get_filing_content_by_accession",
        json={
            "identifier": identifier,
            "accession_number": accession_number,
        },
    )
    return resp.json()


@mcp.tool()
async def analyze_8k(identifier: str, accession_number: str) -> Any:
    """Analyze an 8-K filing for specific events and items."""
    resp = await http_client.post(
        "/analyze_8k",
        json={
            "identifier": identifier,
            "accession_number": accession_number,
        },
    )
    return resp.json()


@mcp.tool()
async def get_filing_sections(identifier: str, accession_number: str) -> Any:
    resp = await http_client.post(
        "/get_filing_sections",
        json={
            "identifier": identifier,
            "accession_number": accession_number,
        },
    )
    return resp.json()


@mcp.tool()
async def get_financials(identifier: str, accession_number: str) -> Any:
    resp = await http_client.post(
        "/get_financials",
        json={
            "identifier": identifier,
            "accession_number": accession_number,
        },
    )
    return resp.json()


@mcp.tool()
async def get_segment_data(identifier: str, accession_number: str) -> Any:
    resp = await http_client.post(
        "/get_segment_data",
        json={
            "identifier": identifier,
            "accession_number": accession_number,
        },
    )
    return resp.json()


@mcp.tool()
async def answer(final_answer: str) -> str:
    await http_client.post("/answer", json={"final_answer": final_answer})
    return f"Answer submitted: {final_answer}"


@mcp.tool()
async def evaluate(rubric: list[dict[str, str | float]]) -> EvaluationResult:
    try:
        resp = await http_client.post("/evaluate", json={"rubric": rubric})
        resp.raise_for_status()
        return EvaluationResult(**resp.json())
    except Exception as e:
        logging.error(f"Evaluation tool error: {e}")
        return EvaluationResult(
            reward=0.0, done=True, content=f"Evaluation error: {e}", isError=True
        )


if __name__ == "__main__":
    mcp.run()
