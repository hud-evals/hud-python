"""Controller lifecycle hooks."""

from controller import mcp, http_client


@mcp.initialize
async def init():
    """Check if the environment is healthy"""
    if http_client:
        await http_client.get("/health")
    else:
        raise ValueError("http_client is not set")


@mcp.shutdown
async def cleanup():
    """Close the HTTP client"""
    if http_client:
        await http_client.aclose()
