"""Claude provider tools — coding (SSH), computer (RFB), MCP proxy.

Memory + hosted tools (web search, web fetch, tool search) will land here
once their capability clients / hosted-tool plumbing is ported.
"""

from __future__ import annotations

from .base import ClaudeToolSpec
from .coding import CLAUDE_BASH_SPEC, CLAUDE_TEXT_EDITOR_SPEC, ClaudeBashTool, ClaudeTextEditorTool
from .computer import CLAUDE_COMPUTER_SPECS, ClaudeComputerTool
from .mcp_proxy import ClaudeMCPProxyTool

__all__ = [
    "CLAUDE_BASH_SPEC",
    "CLAUDE_COMPUTER_SPECS",
    "CLAUDE_TEXT_EDITOR_SPEC",
    "ClaudeBashTool",
    "ClaudeComputerTool",
    "ClaudeMCPProxyTool",
    "ClaudeTextEditorTool",
    "ClaudeToolSpec",
]
