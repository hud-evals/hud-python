"""Claude provider tools — coding (SSH) and MCP proxy.

Computer-use, memory, and hosted tools will land here once their capability
clients (RFB, hosted-tool plumbing) are ported.
"""

from __future__ import annotations

from .base import ClaudeToolSpec
from .coding import CLAUDE_BASH_SPEC, CLAUDE_TEXT_EDITOR_SPEC, ClaudeBashTool, ClaudeTextEditorTool
from .mcp_proxy import ClaudeMCPProxyTool

__all__ = [
    "CLAUDE_BASH_SPEC",
    "CLAUDE_TEXT_EDITOR_SPEC",
    "ClaudeBashTool",
    "ClaudeMCPProxyTool",
    "ClaudeTextEditorTool",
    "ClaudeToolSpec",
]
