from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

from hud.cli.utils.package_runner import run_package_as_mcp


@pytest.mark.asyncio
@mock.patch("hud.cli.utils.package_runner.FastMCP")
async def test_run_package_as_external_command(MockFastMCP):
    proxy = mock.AsyncMock()
    MockFastMCP.as_proxy.return_value = proxy
    await run_package_as_mcp(["python", "-m", "server"], transport="http", port=9999)
    assert proxy.run_async.awaited


@pytest.mark.asyncio
@mock.patch("hud.cli.utils.package_runner.importlib.import_module")
async def test_run_package_import_module(mock_import):
    server = SimpleNamespace(name="test", run_async=mock.AsyncMock())
    mod = SimpleNamespace(mcp=server)
    mock_import.return_value = mod
    await run_package_as_mcp("module_name", transport="stdio")
    assert server.run_async.awaited


@pytest.mark.asyncio
@mock.patch("hud.cli.utils.package_runner.importlib.import_module")
async def test_run_package_import_missing_attr(mock_import):
    mock_import.return_value = SimpleNamespace()
    with pytest.raises(SystemExit):
        await run_package_as_mcp("module_name", transport="stdio")
