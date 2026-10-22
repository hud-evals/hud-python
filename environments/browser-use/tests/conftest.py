"""Browser service boundaries shared by task and lifecycle tests."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def browser_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVEWEB_EVIDENCE_DIR", str(tmp_path / "research"))
    monkeypatch.syspath_prepend(str(ROOT))
    monkeypatch.delitem(sys.modules, "grader", raising=False)
    spec = importlib.util.spec_from_file_location("browser_test_env", ROOT / "env.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_frame_dir", tmp_path / "frames")
    return module


@pytest.fixture
async def started_browser(browser_env, monkeypatch):
    handlers = {}
    current = SimpleNamespace(
        url="about:blank",
        screenshot=AsyncMock(return_value=b"png"),
        wait_for_timeout=AsyncMock(),
        mouse=SimpleNamespace(click=AsyncMock(), wheel=AsyncMock()),
        keyboard=SimpleNamespace(press=AsyncMock()),
        on=lambda event, callback: handlers.setdefault(event, callback),
    )
    current.main_frame = SimpleNamespace(page=current, url=current.url)

    async def goto(url, **kwargs):
        current.url = current.main_frame.url = url
        handlers["framenavigated"](current.main_frame)

    current.goto = AsyncMock(side_effect=goto)
    current.images = SimpleNamespace(evaluate_all=AsyncMock(return_value=[]))
    current.body = SimpleNamespace(inner_text=AsyncMock(return_value="Browser research"))
    current.locator = lambda selector: current.images if selector == "img" else current.body
    context = SimpleNamespace(pages=[current], on=MagicMock())
    browser = SimpleNamespace(contexts=[context], close=AsyncMock())
    driver = SimpleNamespace(
        chromium=SimpleNamespace(connect_over_cdp=AsyncMock(return_value=browser)),
        stop=AsyncMock(),
    )
    server = SimpleNamespace(serve=AsyncMock(), should_exit=False)
    monkeypatch.setattr(browser_env, "wait_for_desktop", AsyncMock())
    monkeypatch.setattr(browser_env, "listening", AsyncMock())
    monkeypatch.setattr(browser_env, "async_playwright", lambda: SimpleNamespace(start=AsyncMock(return_value=driver)))
    monkeypatch.setattr(browser_env.uvicorn, "Server", lambda config: server)
    with monkeypatch.context() as network:
        sock = MagicMock()
        sock.__enter__.return_value = sock
        sock.getsockname.return_value = ("127.0.0.1", 52123)
        network.setattr(browser_env.socket, "socket", lambda: sock)
        await browser_env.initialize()
    try:
        yield SimpleNamespace(env=browser_env, page=current, browser=browser, driver=driver, server=server)
    finally:
        await browser_env.shutdown()
