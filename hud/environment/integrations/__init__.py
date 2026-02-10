"""Provider integrations - format conversion and framework tools."""

from hud.environment.integrations.adk import ADKMixin
from hud.environment.integrations.anthropic import AnthropicMixin
from hud.environment.integrations.claude_agent_sdk import ClaudeAgentSDKMixin
from hud.environment.integrations.gemini import GeminiMixin
from hud.environment.integrations.langchain import LangChainMixin
from hud.environment.integrations.llamaindex import LlamaIndexMixin
from hud.environment.integrations.openai import OpenAIMixin

__all__ = ["IntegrationsMixin"]


class IntegrationsMixin(
    OpenAIMixin,
    AnthropicMixin,
    ClaudeAgentSDKMixin,
    GeminiMixin,
    LangChainMixin,
    LlamaIndexMixin,
    ADKMixin,
):
    """Combined integration mixin for all providers.

    OpenAI:
        as_openai_chat_tools() - Chat Completions format
        as_openai_responses_tools() - Responses API format
        as_openai_agent_tools() - Agents SDK (requires openai-agents)

    Anthropic/Claude:
        as_claude_tools() - Claude API format
        as_claude_programmatic_tools() - Programmatic tool use
        as_anthropic_runner() - Tool runner (requires anthropic)

    Claude Agent SDK:
        as_claude_agent_mcp_server() - In-process MCP server (requires claude-agent-sdk)
        as_claude_agent_options() - Full ClaudeAgentOptions (requires claude-agent-sdk)

    Google/Gemini:
        as_gemini_tools() - Gemini format
        as_gemini_tool_config() - Tool config

    Google ADK:
        as_adk_tools() - ADK FunctionTool objects (requires google-adk)

    LangChain:
        as_langchain_tools() - StructuredTools (requires langchain-core)

    LlamaIndex:
        as_llamaindex_tools() - FunctionTools (requires llama-index-core)
    """
