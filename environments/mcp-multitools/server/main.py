"""
MCP Server for Multi-Tools environment.
Exposes tools: scratchpad, exa (search/fetch), supabase, linear.
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional

import httpx

from hud.server import MCPServer
from hud.tools.types import EvaluationResult

# Configure logging
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

# MCP server
mcp = MCPServer(name="mcp-multitools")

# Environment server URL (backend)
ENV_SERVER_URL = os.getenv("ENV_SERVER_URL", "http://localhost:8000")

# Shared HTTP client
http_client = httpx.AsyncClient(
    base_url=ENV_SERVER_URL,
    timeout=30.0,
    headers={"User-Agent": "HUD-MCP-MultiTools/1.0"},
)

@mcp.initialize
async def init():
    """Ensure environment server is reachable."""
    await http_client.get("/health")
    logger.info("MCP Multi-Tools server initialized")


@mcp.shutdown
async def cleanup():
    """Clean up HTTP client."""
    await http_client.aclose()


# ===========================================
# Setup & Evaluate Tools
# Called by FRAMEWORK based on task JSON config (setup_tool, evaluate_tool)
# NOT in agent_config.allowed_tools, so agent CANNOT call these
# ===========================================
@mcp.tool()
async def setup(initial_data: Optional[Dict[str, str]] = None) -> str:
    """Reset the environment to initial state.
    
    Args:
        initial_data: Optional dict of key-value pairs to pre-populate in scratchpad.
    """
    payload = {"initial_data": initial_data} if initial_data else {}
    await http_client.post("/setup", json=payload)
    if initial_data:
        return f"Environment reset. Scratchpad pre-loaded with {len(initial_data)} keys."
    return "Environment reset. Scratchpad cleared."


@mcp.tool()
async def evaluate(
    expected_keys: Optional[List[str]] = None,
    expected_values: Optional[Dict[str, str]] = None,
    exact_values: Optional[Dict[str, str]] = None,
    min_value_length: int = 0,
) -> EvaluationResult:
    """Evaluate task completion."""
    resp = await http_client.post("/evaluate", json={
        "expected_keys": expected_keys,
        "expected_values": expected_values,
        "exact_values": exact_values,
        "min_value_length": min_value_length,
    })
    return EvaluationResult(**resp.json())


# ===========================================
# Scratchpad Tools
# ===========================================
@mcp.tool()
async def scratchpad_write(key: str, value: str) -> str:
    """Store a value in the scratchpad memory.
    
    Args:
        key: The key to store the value under.
        value: The value to store.
    """
    resp = await http_client.post("/scratchpad/write", json={"key": key, "value": value})
    data = resp.json()
    if data.get("ok"):
        return f"Stored '{key}' in scratchpad"
    return f"Failed to store: {data}"


@mcp.tool()
async def scratchpad_read(key: str) -> str:
    """Read a value from the scratchpad memory.
    
    Args:
        key: The key to read.
    """
    resp = await http_client.post("/scratchpad/read", json={"key": key})
    data = resp.json()
    if data.get("ok"):
        return data.get("value", "")
    return f"Key '{key}' not found"


@mcp.tool()
async def scratchpad_list() -> List[str]:
    """List all keys in the scratchpad memory."""
    resp = await http_client.get("/scratchpad/list")
    data = resp.json()
    return data.get("keys", [])


@mcp.tool()
async def scratchpad_delete(key: str) -> str:
    """Delete a key from the scratchpad memory.
    
    Args:
        key: The key to delete.
    """
    resp = await http_client.post("/scratchpad/delete", json={"key": key})
    data = resp.json()
    if data.get("ok"):
        return f"Deleted '{key}' from scratchpad"
    return f"Key '{key}' not found"


# ===========================================
# Exa Search Tools
# ===========================================
@mcp.tool()
async def search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search the web using Exa.
    
    Args:
        query: The search query.
        max_results: Maximum number of results (default: 5).
    """
    resp = await http_client.post("/exa/search", json={"query": query, "max_results": max_results})
    if resp.status_code != 200:
        return [{"error": f"Search failed: {resp.text}"}]
    return resp.json()


@mcp.tool()
async def fetch(url: str) -> str:
    """Fetch content from a URL using Exa.
    
    Args:
        url: The URL to fetch content from.
    """
    resp = await http_client.post("/exa/fetch", json={"url": url})
    if resp.status_code != 200:
        return f"Fetch failed: {resp.text}"
    data = resp.json()
    return data.get("content", "No content available")


# ===========================================
# Document Store Tools
# ===========================================
@mcp.tool()
async def list_documents() -> List[Dict[str, Any]]:
    """List all available documents in the document store.
    
    Returns a list of documents with their IDs, filenames, and approximate page counts.
    """
    resp = await http_client.get("/documents/list")
    if resp.status_code != 200:
        return [{"error": f"List failed: {resp.text}"}]
    data = resp.json()
    return data.get("documents", [])


@mcp.tool()
async def read_document(doc_id: str, start_page: int = 1, end_page: Optional[int] = None) -> Dict[str, Any]:
    """Read content from a document in the document store.
    
    Args:
        doc_id: The document ID (filename without extension).
        start_page: The page to start reading from (default: 1). Each page is ~3000 characters.
        end_page: The page to stop reading at (optional). If not provided, reads just start_page.
    
    Returns document content with pagination info.
    """
    resp = await http_client.post("/documents/read", json={
        "doc_id": doc_id, "start_page": start_page, "end_page": end_page,
    })
    if resp.status_code != 200:
        return {"error": f"Read failed: {resp.text}"}
    return resp.json()


@mcp.tool()
async def search_document(doc_id: str, query: str, context_chars: int = 200) -> Dict[str, Any]:
    """Search for text within a specific document.
    
    Args:
        doc_id: The document ID to search in.
        query: The text to search for.
        context_chars: Characters of context to include around each match (default: 200).
    
    Returns matching sections with line numbers and surrounding context.
    
    NOTE: Returns maximum 20 matches.
    """
    resp = await http_client.post("/documents/search", json={
        "doc_id": doc_id, "query": query, "context_chars": context_chars,
    })
    if resp.status_code != 200:
        return {"error": f"Search failed: {resp.text}"}
    return resp.json()


# ===========================================
# GitHub API Tools
# ===========================================
@mcp.tool()
async def github_read_file(owner: str, repo: str, path: str, ref: Optional[str] = None) -> Dict[str, Any]:
    """Read a file from a GitHub repository.
    
    Args:
        owner: Repository owner (username or organization).
        repo: Repository name.
        path: Path to the file within the repository.
        ref: Optional branch, tag, or commit SHA (default: main branch).
    
    Returns the file content and metadata.
    """
    resp = await http_client.post("/github/read_file", json={
        "owner": owner, "repo": repo, "path": path, "ref": ref,
    })
    if resp.status_code != 200:
        return {"error": f"GitHub read failed: {resp.text}"}
    return resp.json()


@mcp.tool()
async def github_list_files(owner: str, repo: str, path: str = "", ref: Optional[str] = None) -> Dict[str, Any]:
    """List files in a GitHub repository directory.
    
    Args:
        owner: Repository owner (username or organization).
        repo: Repository name.
        path: Path to the directory (default: root).
        ref: Optional branch, tag, or commit SHA.
    
    Returns a list of files and directories.
    """
    resp = await http_client.post("/github/list_files", json={
        "owner": owner, "repo": repo, "path": path, "ref": ref,
    })
    if resp.status_code != 200:
        return {"error": f"GitHub list failed: {resp.text}"}
    return resp.json()


@mcp.tool()
async def github_search_code(owner: str, repo: str, query: str, path: Optional[str] = None) -> Dict[str, Any]:
    """Search for code in a GitHub repository.
    
    Note: Requires GITHUB_TOKEN for authentication (rate limited otherwise).
    
    Args:
        owner: Repository owner (username or organization).
        repo: Repository name.
        query: Search query (code to find).
        path: Optional path to limit search scope.
    
    Returns matching files with their paths.
    """
    resp = await http_client.post("/github/search_code", json={
        "owner": owner, "repo": repo, "query": query, "path": path,
    })
    if resp.status_code != 200:
        return {"error": f"GitHub search failed: {resp.text}"}
    return resp.json()


# ===========================================
# Python Code Execution Tool
# ===========================================
@mcp.tool()
async def execute_python(code: str, timeout: int = 30) -> Dict[str, Any]:
    """Execute Python code and return the output.
    
    The code runs in an environment with access to:
    - Standard libraries (json, os, sys, math, etc.)
    - supabase-py (use SUPABASE_URL and SUPABASE_KEY variables)
    - requests, httpx
    
    Args:
        code: Python code to execute. Use print() to output results.
        timeout: Maximum execution time in seconds (default: 30).
    
    Returns stdout, stderr, and return code.
    
    Example to query Supabase:
        from supabase import create_client
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        result = client.table('products').select('*').limit(5).execute()
        print(result.data)
    """
    resp = await http_client.post("/execute_python", json={
        "code": code, "timeout": timeout,
    })
    if resp.status_code != 200:
        return {"error": f"Execution failed: {resp.text}"}
    return resp.json()


# ===========================================
# Supabase Database Tools (Custom - Limited)
# ===========================================
# @mcp.tool()
# async def db_query(table: str, select: str = "*", filters: Optional[Dict[str, Any]] = None, limit: int = 10) -> Dict[str, Any]:
#     """Query data from a Supabase table (simple version).
    
#     Args:
#         table: The table name to query.
#         select: Columns to select (default: "*").
#         filters: Optional filters as {column: value}.
#         limit: Maximum rows to return (default: 10).
#     """
#     resp = await http_client.post("/supabase/query", json={
#         "table": table, "select": select, "filters": filters, "limit": limit,
#     })
#     if resp.status_code != 200:
#         return {"error": f"Query failed: {resp.text}"}
#     return resp.json()


# @mcp.tool()
# async def db_insert(table: str, data: Dict[str, Any]) -> Dict[str, Any]:
#     """Insert data into a Supabase table.
    
#     Args:
#         table: The table name.
#         data: Data to insert as {column: value}.
#     """
#     resp = await http_client.post("/supabase/insert", json={"table": table, "data": data})
#     if resp.status_code != 200:
#         return {"error": f"Insert failed: {resp.text}"}
#     return resp.json()


# @mcp.tool()
# async def db_update(table: str, data: Dict[str, Any], filters: Dict[str, Any]) -> Dict[str, Any]:
#     """Update data in a Supabase table.
    
#     Args:
#         table: The table name.
#         data: Data to update as {column: value}.
#         filters: Filters to select rows as {column: value}.
#     """
#     resp = await http_client.post("/supabase/update", json={"table": table, "data": data, "filters": filters})
#     if resp.status_code != 200:
#         return {"error": f"Update failed: {resp.text}"}
#     return resp.json()


# ===========================================
# Linear Project Management Tools
# ===========================================
@mcp.tool()
async def linear_list_issues(team_id: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
    """List issues from Linear.
    
    Args:
        team_id: Optional team ID to filter issues.
        limit: Maximum issues to return (default: 100).
    """
    resp = await http_client.post("/linear/list_issues", json={"team_id": team_id, "limit": limit})
    if resp.status_code != 200:
        return {"error": f"List failed: {resp.text}"}
    return resp.json()


@mcp.tool()
async def linear_create_issue(title: str, description: str, team_id: str) -> Dict[str, Any]:
    """Create a new issue in Linear.
    
    Args:
        title: The issue title.
        description: The issue description.
        team_id: The team ID to create the issue in.
    """
    resp = await http_client.post("/linear/create_issue", json={
        "title": title, "description": description, "team_id": team_id,
    })
    if resp.status_code != 200:
        return {"error": f"Create failed: {resp.text}"}
    return resp.json()


@mcp.tool()
async def linear_update_issue(issue_id: str, status: Optional[str] = None, title: Optional[str] = None) -> Dict[str, Any]:
    """Update an existing issue in Linear.
    
    Args:
        issue_id: The issue ID to update.
        status: Optional new status (state ID).
        title: Optional new title.
    """
    resp = await http_client.post("/linear/update_issue", json={
        "issue_id": issue_id, "status": status, "title": title,
    })
    if resp.status_code != 200:
        return {"error": f"Update failed: {resp.text}"}
    return resp.json()


if __name__ == "__main__":
    mcp.run()
