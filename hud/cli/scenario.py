"""CLI proxy for scenario operations against a running MCP server.

Persists the MCP session ID to /tmp/.hud_scenario_session so that
setup and grade can run as separate processes (e.g. docker exec).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import typer

from hud.utils.hud_console import HUDConsole

hud_console = HUDConsole()

scenario_app = typer.Typer(
    help="Run scenario operations (list, setup, grade) against a running server",
    rich_markup_mode="rich",
)

DEFAULT_URL = "http://localhost:8080/mcp"
SESSION_FILE = Path("/tmp/.hud_scenario_session")  # noqa


def _save_session_id(session_id: str | None) -> None:
    if session_id:
        SESSION_FILE.write_text(session_id)


def _load_session_id() -> str | None:
    if SESSION_FILE.exists():
        sid = SESSION_FILE.read_text().strip()
        return sid if sid else None
    return None


async def _client(url: str, session_id: str | None = None) -> Any:
    """Create an MCP client, optionally resuming a session."""
    from fastmcp import Client
    from fastmcp.client.transports.http import StreamableHttpTransport

    headers: dict[str, str] = {}
    if session_id:
        headers["mcp-session-id"] = session_id

    transport = StreamableHttpTransport(url, headers=headers)
    client = Client(transport)
    await client.__aenter__()
    return client


def _get_session_id_from_client(client: Any) -> str | None:
    """Extract the MCP session ID from the client's transport."""
    transport = getattr(client, "transport", None)
    if transport and hasattr(transport, "get_session_id"):
        return transport.get_session_id()
    return None


async def _resolve_scenario_name(client: Any, scenario: str) -> str:
    """Resolve a short scenario name to its full env:scenario prompt ID.

    If scenario already contains ':', returns it as-is.
    Otherwise, searches available prompts for a match.
    """
    if ":" in scenario:
        return scenario

    prompts = await client.list_prompts()
    for p in prompts:
        if ":" in p.name and p.name.split(":", 1)[-1] == scenario:
            return p.name

    available = [p.name.split(":", 1)[-1] for p in prompts if ":" in p.name]
    raise typer.Exit(
        hud_console.error(f"Scenario '{scenario}' not found. Available: {', '.join(available)}")
        or 1
    )


def _parse_args(args_json: str | None) -> dict[str, str]:
    if not args_json:
        return {}
    try:
        raw = json.loads(args_json)
    except json.JSONDecodeError as e:
        hud_console.error(f"Invalid JSON: {e}")
        raise typer.Exit(1) from None
    return {k: json.dumps(v) if not isinstance(v, str) else v for k, v in raw.items()}


@scenario_app.command(name="list")
def list_cmd(
    url: str = typer.Option(DEFAULT_URL, "--url", "-u"),
) -> None:
    """List scenarios on the running server."""

    async def _run() -> None:
        client = await _client(url)
        try:
            for p in sorted(await client.list_prompts(), key=lambda x: x.name):
                if ":" not in p.name:
                    continue
                args = ", ".join(a.name for a in (p.arguments or []))
                print(f"  {p.name}({args})" if args else f"  {p.name}")  # noqa: T201
        finally:
            await client.__aexit__(None, None, None)

    asyncio.run(_run())


@scenario_app.command(name="setup")
def setup_cmd(
    scenario: str = typer.Argument(..., help="scenario name (auto-resolves env prefix)"),
    args: str | None = typer.Option(None, "--args", "-a", help="JSON args"),
    url: str = typer.Option(DEFAULT_URL, "--url", "-u"),
) -> None:
    """Run setup, print the prompt."""

    async def _run() -> None:
        client = await _client(url)
        full_name = await _resolve_scenario_name(client, scenario)
        result = await client.get_prompt(full_name, _parse_args(args))
        _save_session_id(_get_session_id_from_client(client))
        for msg in result.messages:
            print(msg.content.text if hasattr(msg.content, "text") else msg.content)  # noqa: T201

    asyncio.run(_run())


@scenario_app.command(name="grade")
def grade_cmd(
    scenario: str = typer.Argument(..., help="scenario name (auto-resolves env prefix)"),
    answer: str = typer.Option("", "--answer", "-A"),
    url: str = typer.Option(DEFAULT_URL, "--url", "-u"),
) -> None:
    """Submit answer and grade, print the result."""

    async def _run() -> None:
        session_id = _load_session_id()
        client = await _client(url, session_id=session_id)
        try:
            full_name = await _resolve_scenario_name(client, scenario)
            short_name = full_name.split(":")[-1]
            await client.call_tool("_hud_submit", {"scenario": short_name, "answer": answer})
            contents = await client.read_resource(full_name)
            first = contents[0] if isinstance(contents, list) else contents
            text = first.text if hasattr(first, "text") else str(first)
            print(json.dumps(json.loads(text)))  # noqa: T201
        finally:
            await client.__aexit__(None, None, None)

    asyncio.run(_run())


@scenario_app.command(name="run")
def run_cmd(
    scenario: str = typer.Argument(..., help="scenario name (auto-resolves env prefix)"),
    args: str | None = typer.Option(None, "--args", "-a", help="JSON args"),
    answer: str = typer.Option("", "--answer", "-A"),
    url: str = typer.Option(DEFAULT_URL, "--url", "-u"),
) -> None:
    """Setup + grade in one shot (for testing graders)."""

    async def _run() -> None:
        client = await _client(url)
        try:
            full_name = await _resolve_scenario_name(client, scenario)
            result = await client.get_prompt(full_name, _parse_args(args))
            for msg in result.messages:
                prompt = msg.content.text if hasattr(msg.content, "text") else str(msg.content)
                hud_console.info(f"Prompt: {prompt}")

            short_name = full_name.split(":")[-1]
            await client.call_tool("_hud_submit", {"scenario": short_name, "answer": answer})
            contents = await client.read_resource(full_name)
            first = contents[0] if isinstance(contents, list) else contents
            text = first.text if hasattr(first, "text") else str(first)
            print(json.dumps(json.loads(text)))  # noqa: T201
        finally:
            await client.__aexit__(None, None, None)

    asyncio.run(_run())
