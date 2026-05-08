from __future__ import annotations

import pytest

from hud.agents.base import CategorizedTools
from hud.agents.claude import (
    ClaudeAgent,
    ClaudeToolSearchTool,
    ClaudeWebFetchTool,
    ClaudeWebSearchTool,
)
from hud.agents.gemini import (
    GeminiAgent,
    GeminiCodeExecutionTool,
    GeminiGoogleSearchTool,
    GeminiUrlContextTool,
)
from hud.agents.openai import (
    OpenAIAgent,
    OpenAICodeInterpreterTool,
    OpenAIToolSearchTool,
)


def test_claude_agent_configured_hosted_tools() -> None:
    agent = ClaudeAgent.create(
        model_client=object(),
        hosted_tools=[
            ClaudeWebSearchTool(max_uses=3),
            ClaudeWebFetchTool(citations_enabled=True),
            ClaudeToolSearchTool(threshold=7),
        ],
    )
    agent._available_tools = []
    agent._categorized_tools = CategorizedTools()

    agent._convert_tools_for_claude()

    assert {tool.get("type") for tool in agent.claude_tools if isinstance(tool, dict)} == {
        "web_search_20250305",
        "web_fetch_20250910",
        "tool_search_tool_bm25_20251119",
    }
    assert agent._required_betas == set()
    assert agent._tool_search_threshold == 7


def test_claude_hosted_domain_filters_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="either allowed_domains or blocked_domains"):
        ClaudeWebSearchTool(
            allowed_domains=["example.com"],
            blocked_domains=["blocked.example"],
        ).to_params()

    with pytest.raises(ValueError, match="either allowed_domains or blocked_domains"):
        ClaudeWebFetchTool(
            allowed_domains=["example.com"],
            blocked_domains=["blocked.example"],
        ).to_params()


def test_openai_agent_configured_hosted_tools() -> None:
    agent = OpenAIAgent.create(
        model_client=object(),
        hosted_tools=[
            OpenAICodeInterpreterTool(container={"type": "auto"}),
            OpenAIToolSearchTool(threshold=4),
        ],
    )
    agent._available_tools = []
    agent._categorized_tools = CategorizedTools()

    agent._convert_tools_for_openai()

    assert {"code_interpreter", "tool_search"} <= {
        tool.get("type") for tool in agent._openai_tools if isinstance(tool, dict)
    }
    assert agent._tool_search_threshold == 4


def test_openai_hosted_tools_are_model_gated() -> None:
    agent = OpenAIAgent.create(
        model_client=object(),
        model="gpt-4.1",
        hosted_tools=[
            OpenAICodeInterpreterTool(container={"type": "auto"}),
            OpenAIToolSearchTool(threshold=4),
        ],
    )
    agent._available_tools = []
    agent._categorized_tools = CategorizedTools()

    agent._convert_tools_for_openai()

    assert agent._openai_tools == []
    assert agent._tool_search_threshold is None


def test_gemini_agent_configured_hosted_tools() -> None:
    agent = GeminiAgent.create(
        model_client=object(),
        hosted_tools=[
            GeminiGoogleSearchTool(),
            GeminiUrlContextTool(),
            GeminiCodeExecutionTool(),
        ],
    )
    agent._available_tools = []
    agent._categorized_tools = CategorizedTools()

    agent._convert_tools_for_gemini()

    assert any(getattr(tool, "google_search", None) is not None for tool in agent.gemini_tools)
    assert any(getattr(tool, "url_context", None) is not None for tool in agent.gemini_tools)
    assert any(getattr(tool, "code_execution", None) is not None for tool in agent.gemini_tools)


def test_gemini_google_search_rejects_unsupported_dynamic_threshold() -> None:
    tool = GeminiGoogleSearchTool(dynamic_threshold=0.2)

    try:
        tool.to_params()
    except ValueError as exc:
        assert "dynamic_threshold" in str(exc)
    else:
        raise AssertionError("dynamic_threshold should be rejected")
