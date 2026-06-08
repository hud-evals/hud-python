"""Provider-facing agent tools.

``AgentTool`` is the abstract base, generic in its client type. Capability
bases — ``SSHTool``, ``MCPTool``, ``RFBTool`` — bind that generic and add
per-protocol helpers. Provider subclasses extend one of those bases.

``HostedTool`` is a separate kind: provider-built-in tools (Claude WebSearch,
Gemini CodeExecution, …) that aren't backed by any capability/client and are
declared by agent config.
"""

from __future__ import annotations

from .base import AgentTool, AgentToolSpec, ClientT, result_text, tool_err, tool_ok
from .hosted import HostedTool
from .mcp import MCPTool
from .rfb import RFBTool
from .ssh import SSHTool

__all__ = [
    "AgentTool",
    "AgentToolSpec",
    "ClientT",
    "HostedTool",
    "MCPTool",
    "RFBTool",
    "SSHTool",
    "result_text",
    "tool_err",
    "tool_ok",
]
