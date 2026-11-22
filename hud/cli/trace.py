"""``hud trace`` — render a rollout's conversation turns."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
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
from hud.telemetry.span import (
    PAYLOAD_ATTRIBUTE,
    SCHEMA_ATTRIBUTE,
    TRACE_SCHEMA,
    normalize_trace_id,
)
from hud.types import STEP_SCHEMA, Step, TextContent
from hud.utils.deprecation import warn_deprecated
from hud.utils.exceptions import HudRequestError
from hud.utils.hud_console import HUDConsole
from hud.utils.platform import PlatformClient

if TYPE_CHECKING:
    from pathlib import Path

hud_console = HUDConsole()


class _Events(BaseModel):
    events: list[dict[str, Any]]
    status: str
    reward: float | None = None


class _TraceGroup(CLIGroup):
    def resolve_command(self, ctx: Any, args: list[str]) -> tuple[Any, Any, list[str]]:
        try:
            UUID(args[0])
        except ValueError:
            return super().resolve_command(ctx, args)
        warn_deprecated("`hud trace <trace-id>`", "use `hud trace get <trace-id>`")
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

    Reads the local trace file (HUD_TRACE_DIR, ~/.hud/traces by default), else
    the HUD platform.

    [not dim]Examples:
        hud trace get <trace-id>
        hud trace get <trace-id> --json[/not dim]
    """
    otel_id = normalize_trace_id(trace_id)
    local = settings.trace_dir / f"{otel_id}.jsonl" if settings.trace_dir else None

    events: list[dict[str, Any]]
    if local is not None and local.exists():
        source = f"local ({local})"
        events = _local_events(local)
    else:
        source = "platform"
        try:
            trace = PlatformClient.from_settings().get(f"/trace/{trace_id}/events", _Events)
        except HudRequestError as exc:
            raise CliError.from_http(exc, resource="Trace", input={"trace_id": trace_id}) from exc
        events = [
            *trace.events,
            {"kind": "outcome", "status": trace.status, "reward": trace.reward},
        ]

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
        elif kind == "outcome":
            hud_console.stdout.print(Rule("[cyan]Outcome[/cyan]", style="cyan"))
            hud_console.stdout.print(
                Text(f"{event.get('status')}, reward {event.get('reward')}", style="bold")
            )
            if event.get("error"):
                hud_console.stdout.print(Text(f"  error: {event['error']}", style="red"))

    web = settings.hud_web_url.rstrip("/")
    hud_console.stdout.print(f"\n[dim]View: {web}/trace/{UUID(otel_id)}[/dim]")
    return events


def _local_events(path: Path) -> list[dict[str, Any]]:
    """A local trace file as the platform's ``/trace/{id}/events`` events, then its outcome."""
    spans: list[dict[str, Any]] = []
    incomplete = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            spans.append(json.loads(line))
        except json.JSONDecodeError:
            incomplete += 1  # the exporter appends; an interrupted write leaves a partial line
    if incomplete:
        hud_console.warning(f"Skipped {incomplete} incomplete span record(s) in {path}")

    events: list[dict[str, Any]] = []
    outcome: dict[str, Any] | None = None
    for span in sorted(spans, key=lambda s: s.get("start_time", "")):
        attrs = span.get("attributes", {})
        payload = attrs.get(PAYLOAD_ATTRIBUTE)
        if not isinstance(payload, dict):
            continue
        schema = attrs.get(SCHEMA_ATTRIBUTE)
        if schema == TRACE_SCHEMA:
            outcome = {
                "kind": "outcome",
                "status": payload.get("status"),
                "reward": payload.get("reward"),
                "error": payload.get("error"),
            }
        elif schema == STEP_SCHEMA and (event := _step_event(payload, span.get("name"))):
            events.append(
                {
                    "id": span.get("span_id"),
                    "started_at": payload.get("started_at") or span.get("start_time"),
                    "ended_at": payload.get("ended_at") or span.get("end_time"),
                    **event,
                }
            )
    return [*events, outcome] if outcome is not None else events


def _step_event(payload: dict[str, Any], span_name: str | None) -> dict[str, Any] | None:
    """One step as the platform projects it; ``None`` for one it shows only as a raw event."""
    match payload.get("source"):
        case "user":
            messages = Step.model_validate(payload).messages
            texts = [m.content.text for m in messages if isinstance(m.content, TextContent)]
            return {"kind": "user_message", "text": "\n\n".join(texts)}
        case "agent":
            agent = AgentStep.model_validate(payload)
            return {
                "kind": "agent_message",
                "text": agent.content,
                "reasoning": agent.reasoning,
                "finish_reason": agent.finish_reason,
                "refusal": agent.refusal,
                "tool_calls": [
                    {"tool_call_id": c.id, "name": c.name, "arguments": c.arguments or {}}
                    for c in agent.tool_calls
                ],
            }
        case "tool":
            tool = ToolStep.model_validate(payload)
            call, result = tool.call, tool.result
            blocks = result.content if result else []
            text = "\n".join(block.text for block in blocks if isinstance(block, TextContent))
            failed = bool(result and result.isError)
            return {
                "kind": "tool_call",
                "method": "tools/call",
                "tool_call_id": call.id if call else None,
                "tool_name": call.name if call else None,
                "arguments": (call.arguments if call else None) or {},
                "result_text": None if failed else text or None,
                "error": tool.error or ((text or "tool error (no message)") if failed else None),
            }
        case "task":
            step = Step.model_validate(payload)
            task = step.task_call
            return {
                "kind": f"scenario_{task.phase if task else 'setup'}",
                "name": task.name if task else span_name,
                "arguments": task.arguments if task and isinstance(task.arguments, dict) else {},
                "result": task.result if task else None,
                "error": step.error,
            }
    return None
