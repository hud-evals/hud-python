from __future__ import annotations

import pytest
from fastmcp import FastMCP

from hud.server.router import HiddenRouter


def _hidden_router() -> HiddenRouter:
    router = FastMCP("inner")

    @router.tool()
    async def echo(value: str = "default") -> str:
        return value

    return HiddenRouter("dispatch", router=router)


async def test_hidden_router_dispatch_accepts_json_object_arguments() -> None:
    hidden = _hidden_router()
    tool = await hidden._local_provider.get_tool("dispatch")

    assert tool is not None
    result = await tool.run({"name": "echo", "arguments": '{"value": "ok"}'})

    assert getattr(result, "structured_content", {}).get("result") == "ok"


async def test_hidden_router_dispatch_rejects_malformed_json_arguments() -> None:
    hidden = _hidden_router()
    tool = await hidden._local_provider.get_tool("dispatch")

    assert tool is not None
    with pytest.raises(ValueError, match="Invalid JSON arguments"):
        await tool.run({"name": "echo", "arguments": "{"})


async def test_hidden_router_dispatch_rejects_non_object_json_arguments() -> None:
    hidden = _hidden_router()
    tool = await hidden._local_provider.get_tool("dispatch")

    assert tool is not None
    with pytest.raises(TypeError, match="must decode to an object"):
        await tool.run({"name": "echo", "arguments": '["not", "an", "object"]'})
