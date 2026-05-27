"""OpenAI provider tools."""

from __future__ import annotations

from .base import OpenAIToolSpec
from .coding import OPENAI_SHELL_SPEC, OpenAIShellTool
from .computer import OPENAI_COMPUTER_SPEC, OpenAIComputerTool
from .mcp_proxy import OpenAIMCPProxyTool

__all__ = [
    "OPENAI_COMPUTER_SPEC",
    "OPENAI_SHELL_SPEC",
    "OpenAIComputerTool",
    "OpenAIMCPProxyTool",
    "OpenAIShellTool",
    "OpenAIToolSpec",
]
