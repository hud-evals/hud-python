"""``hud trace <trace_id>`` — render a rollout's conversation turns."""

from __future__ import annotations

import contextlib
import json
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

console = Console()

trace_app = typer.Typer(
    name="trace",
    help="Inspect a rollout trace",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


@trace_app.callback(invoke_without_command=True)
def trace_command(
    ctx: typer.Context,
    trace_id: str = typer.Argument(..., help="Trace ID (UUID or 32-hex OTel id)"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON"),
    local_dir: str | None = typer.Option(
        None, "--local-dir", help="Override HUD_TELEMETRY_LOCAL_DIR"
    ),
) -> None:
    """Render the turns and tool calls for one rollout.

    Checks ``HUD_TELEMETRY_LOCAL_DIR`` first (fast, no API needed), then
    falls back to ``GET /v2/trace/{id}/events`` on the platform.
    """
    if ctx.invoked_subcommand is not None:
        return

    from hud.cli.utils.api import require_api_key
    from hud.settings import settings
    from hud.telemetry.span import normalize_trace_id

    dir_to_use = local_dir or settings.telemetry_local_dir
    otel_id = normalize_trace_id(trace_id)

    events: list[dict[str, Any]] | None = None
    source = "platform"

    if dir_to_use:
        from pathlib import Path

        path = Path(dir_to_use) / f"{otel_id}.jsonl"
        if path.exists():
            events = _load_local(path)
            source = f"local ({path})"

    if events is None:
        require_api_key("fetch trace")
        events = _load_remote(trace_id)

    if json_output:
        console.print_json(json.dumps(events, indent=2, default=str))
        return

    if not events:
        console.print("[yellow]No events found for this trace.[/yellow]")
        return

    console.print(
        Panel.fit(f"[bold cyan]Trace[/bold cyan] [dim]{trace_id}[/dim]", border_style="cyan")
    )
    console.print(f"[dim]Source: {source}[/dim]\n")
    _render_events(events)

    web = settings.hud_web_url.rstrip("/")
    console.print(f"\n[dim]View: {web}/trace/{trace_id}[/dim]")


# ── local JSONL ────────────────────────────────────────────────────────────────


def _load_local(path: Any) -> list[dict[str, Any]]:
    """Read spans from a local .jsonl file and convert to a flat event list."""
    spans: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                with contextlib.suppress(json.JSONDecodeError):
                    spans.append(json.loads(line))

    events: list[dict[str, Any]] = []
    for span in sorted(spans, key=lambda s: s.get("start_time", "")):
        attrs = span.get("attributes", {})
        schema = attrs.get("hud.schema")
        payload = attrs.get("hud.payload", {})
        if not isinstance(payload, dict):
            continue

        source = payload.get("source")
        if schema == "hud.step.v1":
            if source == "agent":
                events.append(
                    {
                        "kind": "agent_message",
                        "text": payload.get("content"),
                        "reasoning": payload.get("reasoning"),
                        "tool_calls": payload.get("tool_calls") or [],
                        "error": payload.get("error"),
                    }
                )
            elif source == "tool":
                # ToolStep messages hold the tool results
                events.extend(
                    {
                        "kind": "tool_result",
                        "name": msg.get("name") or msg.get("tool_call_id"),
                        "result": _msg_text(msg),
                    }
                    for msg in payload.get("messages", [])
                    if msg.get("role") == "tool"
                )
    return events


def _msg_text(msg: dict[str, Any]) -> str:
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"
        ]
        return "\n".join(parts)
    return str(content)


# ── remote events API ──────────────────────────────────────────────────────────


def _load_remote(trace_id: str) -> list[dict[str, Any]]:
    from hud.utils.platform import PlatformClient

    client = PlatformClient.from_settings()
    try:
        data = client.get(f"/trace/{trace_id}/events")
    except Exception as e:
        console.print(f"[red]Failed to fetch trace: {e}[/red]")
        raise typer.Exit(1) from e

    if isinstance(data, dict):
        return data.get("events", [])
    return []


# ── rendering ──────────────────────────────────────────────────────────────────


def _render_events(events: list[dict[str, Any]]) -> None:
    turn = 0
    for ev in events:
        kind = ev.get("kind")

        if kind == "agent_message":
            turn += 1
            console.print(Rule(f"[cyan]Turn {turn} — agent[/cyan]", style="cyan"))

            reasoning = ev.get("reasoning")
            if reasoning:
                console.print(Text(reasoning, style="dim italic"))

            text = ev.get("text")
            if text:
                console.print(text)

            for tc in ev.get("tool_calls") or []:
                name = tc.get("name") or tc.get("function", {}).get("name", "?")
                args = tc.get("arguments") or tc.get("function", {}).get("arguments") or {}
                if isinstance(args, str):
                    with contextlib.suppress(Exception):
                        args = json.loads(args)
                console.print(
                    f"  [green]→[/green] [bold]{name}[/bold]({_fmt_args(args)})",
                    highlight=False,
                )

            if ev.get("error"):
                console.print(f"  [red]error: {ev['error']}[/red]")

        elif kind in ("tool_call", "tool_result"):
            name = ev.get("tool_name") or ev.get("name") or "?"
            result = ev.get("result_text") or ev.get("result") or ""
            error = ev.get("error")
            if error:
                console.print(f"  [red]✗ {name}: {error}[/red]")
            else:
                console.print(f"  [dim]{name} →[/dim]")
                for line in str(result).splitlines():
                    console.print(f"    {line}")

        elif kind == "environment":
            msg = ev.get("text") or ev.get("content") or ""
            if msg:
                console.print(Rule("[yellow]env[/yellow]", style="yellow"))
                console.print(msg)


def _fmt_args(args: Any) -> str:
    if isinstance(args, dict):
        return ", ".join(f"{k}={v!r}" for k, v in args.items())
    return str(args)
