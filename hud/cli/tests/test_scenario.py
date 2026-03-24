"""Integration tests for hud scenario CLI — real environments, real MCP calls."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any

import httpx
import pytest

from hud.environment import Environment


def _make_env() -> Environment:
    from hud.tools.types import EvaluationResult, SubScore

    env = Environment("test-env")

    @env.scenario(name="echo")
    async def echo(message: str = "hello"):
        yield f"Repeat this back exactly: {message}"
        yield 1.0

    @env.scenario(name="add")
    async def add(a: int = 1, b: int = 2):
        answer = yield f"What is {a} + {b}?"
        try:
            result = int(answer) if isinstance(answer, str) else 0
            yield 1.0 if result == a + b else 0.0
        except (ValueError, TypeError):
            yield 0.0

    @env.scenario(name="multi_check")
    async def multi_check(target: str = "apple"):
        answer = yield f"Name a fruit. Target: {target}"
        score = 0.0
        mentioned = target.lower() in (answer or "").lower()
        is_fruit = any(f in (answer or "").lower() for f in ["apple", "banana", "orange"])
        if mentioned:
            score += 0.6
        if is_fruit:
            score += 0.4
        yield EvaluationResult(
            reward=score,
            done=True,
            content=f"mentioned={mentioned}, is_fruit={is_fruit}",
            subscores=[
                SubScore(name="mentioned_target", weight=0.6, value=1.0 if mentioned else 0.0),
                SubScore(name="is_fruit", weight=0.4, value=1.0 if is_fruit else 0.0),
            ],
        )

    return env


TEST_PORT = 18932


def _text(content: Any) -> str:
    return getattr(content, "text", str(content))


def _resource_text(contents: Any) -> str:
    first = contents[0] if isinstance(contents, list) else contents
    return getattr(first, "text", str(first))


@pytest.fixture(scope="module")
def server_url() -> str:
    return f"http://localhost:{TEST_PORT}/mcp"


@pytest.fixture(scope="module", autouse=True)
def _run_server(server_url: str) -> Any:
    """Start the Environment as an HTTP MCP server in a background thread."""
    env = _make_env()
    loop = asyncio.new_event_loop()

    async def _serve() -> None:
        await env.run_async(
            transport="http",
            port=TEST_PORT,
            path="/mcp",
            host="127.0.0.1",
            show_banner=False,
            log_level="ERROR",
        )

    thread = threading.Thread(target=loop.run_until_complete, args=(_serve(),), daemon=True)
    thread.start()

    for _ in range(30):
        try:
            httpx.get(f"http://localhost:{TEST_PORT}/mcp", timeout=1.0)
            break
        except Exception:
            time.sleep(0.2)

    yield server_url

    loop.call_soon_threadsafe(loop.stop)


@pytest.mark.asyncio
async def test_list_scenarios(server_url: str) -> None:
    from fastmcp import Client

    async with Client(server_url) as client:
        prompts = await client.list_prompts()
        scenario_names = [p.name.split(":", 1)[-1] for p in prompts if ":" in p.name]

    assert "echo" in scenario_names
    assert "add" in scenario_names


@pytest.mark.asyncio
async def test_setup_returns_prompt(server_url: str) -> None:
    from fastmcp import Client

    async with Client(server_url) as client:
        result = await client.get_prompt("test-env:echo", {"message": "hi there"})

    assert result.messages
    assert "hi there" in _text(result.messages[0].content)


@pytest.mark.asyncio
async def test_setup_grade_echo(server_url: str) -> None:
    from fastmcp import Client

    async with Client(server_url) as client:
        result = await client.get_prompt("test-env:echo", {"message": "test"})
        assert "test" in _text(result.messages[0].content)

        await client.call_tool("_hud_submit", {"scenario": "echo", "answer": "test"})
        contents = await client.read_resource("test-env:echo")

    data = json.loads(_resource_text(contents))
    assert data["reward"] == 1.0
    assert data["done"] is True


@pytest.mark.asyncio
async def test_setup_grade_add_correct(server_url: str) -> None:
    from fastmcp import Client

    async with Client(server_url) as client:
        result = await client.get_prompt("test-env:add", {"a": "3", "b": "7"})
        prompt = _text(result.messages[0].content)
        assert "3" in prompt
        assert "7" in prompt

        await client.call_tool("_hud_submit", {"scenario": "add", "answer": "10"})
        contents = await client.read_resource("test-env:add")

    data = json.loads(_resource_text(contents))
    assert data["reward"] == 1.0
    assert data["done"] is True


@pytest.mark.asyncio
async def test_setup_grade_add_wrong(server_url: str) -> None:
    from fastmcp import Client

    async with Client(server_url) as client:
        await client.get_prompt("test-env:add", {"a": "5", "b": "5"})
        await client.call_tool("_hud_submit", {"scenario": "add", "answer": "11"})
        contents = await client.read_resource("test-env:add")

    data = json.loads(_resource_text(contents))
    assert data["reward"] == 0.0
    assert data["done"] is True


@pytest.mark.asyncio
async def test_setup_grade_add_invalid_answer(server_url: str) -> None:
    from fastmcp import Client

    async with Client(server_url) as client:
        await client.get_prompt("test-env:add", {"a": "2", "b": "3"})
        await client.call_tool("_hud_submit", {"scenario": "add", "answer": "not a number"})
        contents = await client.read_resource("test-env:add")

    data = json.loads(_resource_text(contents))
    assert data["reward"] == 0.0
    assert data["done"] is True


@pytest.mark.asyncio
async def test_multi_check_full_match(server_url: str) -> None:
    """Correct answer gets full reward with subscores."""
    from fastmcp import Client

    async with Client(server_url) as client:
        result = await client.get_prompt("test-env:multi_check", {"target": "banana"})
        assert "banana" in _text(result.messages[0].content)

        await client.call_tool("_hud_submit", {"scenario": "multi_check", "answer": "banana"})
        contents = await client.read_resource("test-env:multi_check")

    data = json.loads(_resource_text(contents))
    assert data["reward"] == 1.0
    assert data["done"] is True
    assert data["content"] == "mentioned=True, is_fruit=True"
    assert len(data["subscores"]) == 2
    assert data["subscores"][0]["name"] == "mentioned_target"
    assert data["subscores"][0]["value"] == 1.0
    assert data["subscores"][1]["name"] == "is_fruit"
    assert data["subscores"][1]["value"] == 1.0


@pytest.mark.asyncio
async def test_multi_check_partial(server_url: str) -> None:
    """Wrong fruit gets partial reward (is_fruit but not mentioned_target)."""
    from fastmcp import Client

    async with Client(server_url) as client:
        await client.get_prompt("test-env:multi_check", {"target": "banana"})
        await client.call_tool("_hud_submit", {"scenario": "multi_check", "answer": "orange"})
        contents = await client.read_resource("test-env:multi_check")

    data = json.loads(_resource_text(contents))
    assert data["reward"] == pytest.approx(0.4)
    assert data["done"] is True
    assert data["subscores"][0]["value"] == 0.0  # didn't mention target
    assert data["subscores"][1]["value"] == 1.0  # but it's a fruit


@pytest.mark.asyncio
async def test_multi_check_zero(server_url: str) -> None:
    """Completely wrong answer gets zero."""
    from fastmcp import Client

    async with Client(server_url) as client:
        await client.get_prompt("test-env:multi_check", {"target": "banana"})
        await client.call_tool("_hud_submit", {"scenario": "multi_check", "answer": "chair"})
        contents = await client.read_resource("test-env:multi_check")

    data = json.loads(_resource_text(contents))
    assert data["reward"] == 0.0
    assert data["subscores"][0]["value"] == 0.0
    assert data["subscores"][1]["value"] == 0.0


@pytest.mark.asyncio
async def test_cross_session_with_session_id(server_url: str) -> None:
    """Setup and grade in separate client connections using session ID persistence."""
    from fastmcp import Client
    from fastmcp.client.transports.http import StreamableHttpTransport

    from hud.cli.scenario import _get_session_id_from_client

    # Setup — first connection (no __aexit__ to keep session alive)
    transport1 = StreamableHttpTransport(server_url)
    client1 = Client(transport1)
    await client1.__aenter__()
    await client1.get_prompt("test-env:add", {"a": "4", "b": "6"})
    session_id = _get_session_id_from_client(client1)
    assert session_id is not None
    # Skip __aexit__ — keeps session alive on server

    # Grade — second connection resuming the session
    transport2 = StreamableHttpTransport(server_url, headers={"mcp-session-id": session_id})
    client2 = Client(transport2)
    async with client2:
        await client2.call_tool("_hud_submit", {"scenario": "add", "answer": "10"})
        contents = await client2.read_resource("test-env:add")

    data = json.loads(_resource_text(contents))
    assert data["reward"] == 1.0
    assert data["done"] is True


@pytest.mark.asyncio
async def test_grade_without_setup_fails(server_url: str) -> None:
    from fastmcp import Client

    async with Client(server_url) as client:
        with pytest.raises(Exception):
            await client.read_resource("test-env:echo")
