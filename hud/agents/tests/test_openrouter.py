from __future__ import annotations

import pytest

from types import SimpleNamespace
from typing import Any

def _import_agents():
    import mcp.types as types
    from hud.agents.glm45v import Glm45vAgent
    from hud.agents.openrouter import OpenRouterAgent
    from hud.types import MCPToolResult
    return Glm45vAgent, OpenRouterAgent, MCPToolResult, types


def test_openrouter_agent_defaults_to_glm45v() -> None:
    Glm45vAgent, OpenRouterAgent, _, _ = _import_agents()
    agent = OpenRouterAgent()
    assert isinstance(agent._adapter, Glm45vAgent)
    assert agent.model_name == "openrouter/z-ai/glm-4.5v"


def test_openrouter_agent_normalizes_alias() -> None:
    _, OpenRouterAgent, _, _ = _import_agents()
    agent = OpenRouterAgent(model_name="Z-AI/GLM-4.5V")
    assert agent.model_name == "openrouter/z-ai/glm-4.5v"


def test_openrouter_agent_rejects_unknown_model() -> None:
    _, OpenRouterAgent, _, _ = _import_agents()
    with pytest.raises(ValueError):
        OpenRouterAgent(model_name="unknown/model")


@pytest.mark.asyncio
async def test_openrouter_agent_parses_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    Glm45vAgent, OpenRouterAgent, MCPToolResult, types = _import_agents()
    png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO61uFYAAAAASUVORK5CYII="

    async def fake_completion(*_: Any, **__: Any) -> Any:
        message = SimpleNamespace(content=(
            "I will click the button.\n"
            "<|begin_of_box|>{\"type\": \"click\", \"start_box\": [100, 200]}<|end_of_box|>\n"
            "Memory:[]"
        ), reasoning_content=None)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])

    monkeypatch.setattr("hud.agents.glm45v.litellm.acompletion", fake_completion)

    agent = OpenRouterAgent(model_name="z-ai/glm-4.5v")

    messages: list[dict[str, Any]] = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "click the highlighted cell"}],
        },
        {
            "type": "computer_call_output",
            "call_id": "initial",
            "output": {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{png_base64}",
            },
        },
    ]

    response = await agent.get_response(list(messages))

    assert not response.done
    assert response.tool_calls, "expected at least one tool call"

    tool_call = response.tool_calls[0]
    assert tool_call.name == "openai_computer"
    assert tool_call.arguments["type"] == "click"
    # coordinates are normalized from the 1x1 PNG back to pixel space -> 0/0
    assert tool_call.arguments["x"] == 0
    assert tool_call.arguments["y"] == 0

    tool_result = MCPToolResult(
        content=[
            types.ImageContent(type="image", data=png_base64, mimeType="image/png"),
            types.TextContent(type="text", text="button pressed"),
        ]
    )

    rendered = await agent.format_tool_results([tool_call], [tool_result])

    assert any(item.get("type") == "computer_call_output" for item in rendered)
    assert any(
        item.get("type") == "message" and item.get("role") == "user"
        for item in rendered
    )
