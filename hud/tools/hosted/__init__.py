"""Hosted tools that are executed by the provider, not the client.

These tools are declared in the environment but executed server-side by the LLM provider.
The client only declares them and processes the response metadata.

Usage:
    from hud.tools.hosted import GoogleSearchTool, WebSearchTool, WebFetchTool
"""

from hud.tools.hosted.base import HostedTool
from hud.tools.hosted.code_execution import CodeExecutionTool
from hud.tools.hosted.google_search import GoogleSearchTool
from hud.tools.hosted.url_context import UrlContextTool
from hud.tools.hosted.web_fetch import WebFetchTool
from hud.tools.hosted.web_search import WebSearchTool

__all__ = [
    "CodeExecutionTool",
    "GoogleSearchTool",
    "HostedTool",
    "UrlContextTool",
    "WebFetchTool",
    "WebSearchTool",
]
