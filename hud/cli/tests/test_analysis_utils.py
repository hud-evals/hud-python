from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hud.cli.utils.analysis import analyze_environment


@pytest.mark.asyncio
async def test_analyze_environment_returns_build_ready_shape() -> None:
    client = MagicMock()
    client.list_tools = AsyncMock(
        return_value=[
            SimpleNamespace(
                name="setup",
                description="Calls internal functions.",
                inputSchema={"type": "object"},
            )
        ]
    )
    client.read_resource = AsyncMock(return_value=[SimpleNamespace(text='["prepare", "seed"]')])
    client.list_resources = AsyncMock(return_value=[])
    client.list_prompts = AsyncMock(return_value=[])

    analysis = await analyze_environment(client, server_name="local", initialize_ms=321)

    assert analysis["initializeMs"] == 321
    assert analysis["toolCount"] == 1
    assert analysis["internalToolCount"] == 2
    assert analysis["hubTools"] == {"setup": ["prepare", "seed"]}
    assert analysis["success"] is True
    assert analysis["metadata"] == {"initialized": True, "servers": ["local"]}
    assert analysis["prompts"] == []
    assert analysis["resources"] == []
    assert analysis["scenarios"] == []
    assert analysis["tools"][0]["internalTools"] == ["prepare", "seed"]
