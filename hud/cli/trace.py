"""``hud trace`` — render a rollout's conversation turns."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import UUID

import typer
from pydantic import BaseModel
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from hud.agents.types import AgentStep, ToolStep
from hud.cli import (
    CLI,
    CliError,
    CLIGroup,
)
from hud.settings import settings
from hud.telemetry.span import normalize_trace_id
from hud.types import TextContent
from hud.utils.exceptions import HudRequestError
from hud.utils.hud_console import HUDConsole
from hud.utils.platform import PlatformClient

hud_console = HUDConsole()


class _Events(BaseModel):
    events: list[dict[str, Any]]


class _TraceGroup(CLIGroup):
    def resolve_command(self, ctx: Any, args: list[str]) -> tuple[Any, Any, list[str]]:
        try:
            UUID(args[0])
        except ValueError:
            return super().resolve_command(ctx, args)
        return "get", self.commands["get"], args


trace_app = CLI(
    cls=_TraceGroup,
    name="trace",
    help="Inspect a rollout trace.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


@trace_app.command("get")
def get_command(
    trace_id: str = typer.Argument(..., help="Trace ID (UUID or 32-hex OTel id)"),
) -> Any:
    """Render the turns and tool calls for one rollout.

    Checks the local span directory first (``HUD_TELEMETRY_LOCAL_DIR``, or
    ``~/.hud/spans`` when uploads are disabled), then ``GET /v2/trace/{id}/events``.

    [not dim]Examples:
        hud trace get <trace-id>
        hud trace get <trace-id> --json[/not dim]
    """
    otel_id = normalize_trace_id(trace_id)
    local = Path(settings.span_dir) / f"{otel_id}.jsonl" if settings.span_dir else None

    events: list[dict[str, Any]] = []
    if local is not None and local.exists():
        # Local spans: one JSON span per line; step spans carry the conversation payload.
        source = f"local ({local})"
        spans: list[dict[str, Any]] = []
        incomplete = 0
        for line in local.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                spans.append(json.loads(line))
            except json.JSONDecodeError:
                incomplete += 1  # the exporter appends; an interrupted write leaves a partial line
        if incomplete:
            hud_console.warning(f"Skipped {incomplete} incomplete span record(s) in {local}")
        for span in sorted(spans, key=lambda s: s.get("start_time", "")):
            attrs = span.get("attributes", {})
            payload = attrs.get("hud.payload", {})
            if attrs.get("hud.schema") != "hud.step.v1" or not isinstance(payload, dict):
                continue
            if payload.get("source") == "agent":
                step = AgentStep.model_validate(payload)
                events.append(
                    {
                        "kind": "agent_message",
                        "text": step.content,
                        "reasoning": step.reasoning,
                        "tool_calls": [call.model_dump(mode="json") for call in step.tool_calls],
                        "error": step.error,
                    }
                )
            elif payload.get("source") == "tool":
                tool = ToolStep.model_validate(payload)
                result = tool.result
                text = "\n".join(
                    block.text
                    for block in (result.content if result is not None else [])
                    if isinstance(block, TextContent)
                )
                events.append(
                    {
                        "kind": "tool_call",
                        "tool_name": tool.call.name if tool.call is not None else None,
                        "arguments": tool.call.arguments if tool.call is not None else {},
                        "result_text": text,
                        "error": tool.error
                        or (text if result is not None and result.isError else None),
                    }
                )
    else:
        source = "platform"
        try:
            events = PlatformClient.from_settings().get(f"/trace/{trace_id}/events", _Events).events
        except HudRequestError as exc:
            raise CliError.from_http(exc, resource="Trace", input={"trace_id": trace_id}) from exc

    if not events:
        hud_console.stdout.print("[yellow]No events found for this trace.[/yellow]")
        return events
    hud_console.stdout.print(
        Panel.fit(f"[bold cyan]Trace[/bold cyan] [dim]{trace_id}[/dim]", border_style="cyan")
    )
    hud_console.stdout.print(f"[dim]Source: {source}[/dim]\n")

    # Payloads render as Text, never as markup: rich would read a literal
    # `[len(s) // 2]` in agent output as a style tag and drop it.
    turn = 0
    for event in events:
        kind = event.get("kind")
        if kind == "agent_message":
            turn += 1
            hud_console.stdout.print(Rule(f"[cyan]Turn {turn} — agent[/cyan]", style="cyan"))
            if event.get("reasoning"):
                hud_console.stdout.print(Text(event["reasoning"], style="dim italic"))
            if event.get("text"):
                hud_console.stdout.print(Text(str(event["text"])))
            for call in event.get("tool_calls") or []:
                shown = ", ".join(f"{k}={v!r}" for k, v in (call.get("arguments") or {}).items())
                hud_console.stdout.print(
                    Text.assemble("  ", ("→", "green"), " ", (call["name"], "bold"), f"({shown})")
                )
            if event.get("error"):
                hud_console.stdout.print(Text(f"  error: {event['error']}", style="red"))
        elif kind == "tool_call":
            name = event.get("tool_name") or "?"
            if event.get("error"):
                hud_console.stdout.print(Text(f"  ✗ {name}: {event['error']}", style="red"))
            else:
                hud_console.stdout.print(Text(f"  {name} →", style="dim"))
                for line in (event.get("result_text") or "").splitlines():
                    hud_console.stdout.print(Text(f"    {line}"))

    web = settings.hud_web_url.rstrip("/")
    hud_console.stdout.print(f"\n[dim]View: {web}/trace/{UUID(otel_id)}[/dim]")
    return events
