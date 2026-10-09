"""``hud trace``: a rollout's turns, from local spans or the platform."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import pytest
from inline_snapshot import snapshot

from hud import Environment
from hud.agents.openai_compatible import OpenAIChatAgent
from hud.agents.types import OpenAIChatConfig
from hud.eval import LocalRuntime, Task, rollout
from tests.harness import call, say, scrub

from .conftest import API_KEY, WEB_URL

if TYPE_CHECKING:
    from pathlib import Path

    from tests.harness import FakeServices, Hud, HudEnv, Models

TRACE_ID = "03dd2a73-d3df-4d10-a54a-e3d87c2d530d"
COMPACT_TRACE_ID = "03dd2a73d3df4d10a54ae3d87c2d530d"
EVENTS: list[dict[str, Any]] = [
    {
        "kind": "agent_message",
        "text": "Splitting at [len(s) // 2] and [bold]this[/bold]",
        "reasoning": "Think about the middle.",
        "tool_calls": [{"name": "bash", "arguments": {"command": "ls"}}],
    },
    {"kind": "tool_call", "tool_name": "bash", "result_text": "a.txt\nb.txt"},
    {"kind": "tool_call", "tool_name": "bash", "error": "exit status 1"},
    {"kind": "agent_message", "text": "done", "error": "max steps reached"},
]


@pytest.fixture
def events(platform: FakeServices) -> FakeServices:
    platform.route("api", "GET", "/v2/trace/{id}/events", json={"events": EVENTS})
    return platform


def test_a_platform_trace_renders_turns_and_tool_calls(hud: Hud, events: FakeServices) -> None:
    result = hud("trace", COMPACT_TRACE_ID)

    assert result.exit_code == 0, result
    assert result.lines == snapshot(
        [
            "Trace 03dd2a73d3df4d10a54ae3d87c2d530d",
            "Source: platform",
            "Turn 1 — agent",
            "Think about the middle.",
            "Splitting at [len(s) // 2] and [bold]this[/bold]",
            "→ bash(command='ls')",
            "bash →",
            "a.txt",
            "b.txt",
            "✗ bash: exit status 1",
            "Turn 2 — agent",
            "done",
            "error: max steps reached",
            "View: https://hud.example/trace/03dd2a73-d3df-4d10-a54a-e3d87c2d530d",
        ]
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["trace", TRACE_ID, "--json"],
        ["trace", "--json", TRACE_ID],
        ["trace", "get", TRACE_ID, "--json"],
        ["trace", "get", "--json", TRACE_ID],
    ],
)
def test_trace_json_is_the_event_list(hud: Hud, events: FakeServices, argv: list[str]) -> None:
    result = hud(*argv)

    assert result.exit_code == 0, result
    assert result.json == EVENTS
    (request,) = events.requests("api", "GET", "/v2/trace/{id}/events")
    assert request.params == {"id": TRACE_ID}


@pytest.mark.parametrize(
    ("argv", "exit_code", "document"),
    [
        (
            ["trace", TRACE_ID, "--json"],
            1,
            snapshot(
                {
                    "error": "not_found",
                    "message": "Request failed: missing",
                    "input": {"trace_id": "03dd2a73-d3df-4d10-a54a-e3d87c2d530d"},
                    "suggestion": "Check the trace id, or list existing ones.",
                }
            ),
        ),
        (
            ["trace", "unknown", "--json"],
            2,
            snapshot({"error": "usage", "message": "No such command 'unknown'."}),
        ),
    ],
)
def test_a_missing_or_malformed_trace_is_an_error_document(
    hud: Hud, platform: FakeServices, argv: list[str], exit_code: int, document: dict[str, Any]
) -> None:
    platform.route("api", "GET", "/v2/trace/{id}/events", status=404, json={"detail": "missing"})

    result = hud(*argv)

    assert result.exit_code == exit_code, result
    assert result.json == document


def test_a_trace_without_events_says_so(hud: Hud, platform: FakeServices) -> None:
    platform.route("api", "GET", "/v2/trace/{id}/events", json={"events": []})

    result = hud("trace", TRACE_ID)

    assert (result.exit_code, result.stdout) == (0, "No events found for this trace.\n")


# ─── local spans ────────────────────────────────────────────────────────


async def write_a_report(
    models: Models, hud_env: HudEnv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> str:
    """Run a rollout whose scripted model writes REPORT.md through a tool; return its trace id.

    Spans land in ``spans/`` under ``tmp_path``, the working directory of the
    rollout and of ``hud``, so rendered paths stay short.
    """
    monkeypatch.chdir(tmp_path)
    hud_env.set(HUD_API_KEY=API_KEY, HUD_WEB_URL=WEB_URL, HUD_TELEMETRY_LOCAL_DIR="spans")
    workspace = tmp_path / "workspace"
    report = workspace / "REPORT.md"
    env = Environment("report")
    env.workspace(workspace, guest_path=str(workspace))

    @env.initialize
    async def seed() -> None:
        workspace.mkdir(parents=True, exist_ok=True)

    @env.template()
    async def write_report():
        yield "Write PASS to REPORT.md."
        yield 1.0 if report.read_text().strip() == "PASS" else 0.0

    models.script([call("write", filePath="REPORT.md", content="PASS"), say("Wrote it.")])
    agent = OpenAIChatAgent(OpenAIChatConfig(model="scripted", max_steps=4))
    run = await rollout(Task(env="report", id="write_report"), agent, runtime=LocalRuntime(env))
    assert run.reward == 1.0
    assert run.trace_id is not None
    return run.trace_id


async def test_a_local_trace_lists_tool_calls_with_their_results(
    hud: Hud,
    services: FakeServices,
    models: Models,
    hud_env: HudEnv,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_id = await write_a_report(models, hud_env, tmp_path, monkeypatch)

    result = hud("trace", "get", trace_id, "--json", cwd=tmp_path)

    assert result.exit_code == 0, result
    assert result.lines == snapshot(
        [
            "[",
            "{",
            '"kind": "agent_message",',
            '"text": "",',
            '"reasoning": null,',
            '"tool_calls": [',
            "{",
            '"name": "write",',
            '"arguments": {',
            '"filePath": "REPORT.md",',
            '"content": "PASS"',
            "},",
            '"id": "call_1"',
            "}",
            "],",
            '"error": null',
            "},",
            "{",
            '"kind": "tool_call",',
            '"tool_name": "write",',
            '"arguments": {',
            '"filePath": "REPORT.md",',
            '"content": "PASS"',
            "},",
            '"result_text": "wrote 4 bytes to REPORT.md",',
            '"error": null',
            "},",
            "{",
            '"kind": "agent_message",',
            '"text": "Wrote it.",',
            '"reasoning": null,',
            '"tool_calls": [],',
            '"error": null',
            "}",
            "]",
        ]
    )
    assert services.requests("api") == []


async def test_a_local_trace_renders_and_skips_a_record_cut_short(
    hud: Hud, models: Models, hud_env: HudEnv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace_id = await write_a_report(models, hud_env, tmp_path, monkeypatch)
    spans = tmp_path / "spans" / f"{uuid.UUID(trace_id).hex}.jsonl"
    complete = spans.read_text()
    spans.write_text(complete + complete.splitlines()[-1][:40])

    result = hud("trace", trace_id, cwd=tmp_path)

    assert result.exit_code == 0, result
    assert scrub(result.stdout.replace(uuid.UUID(trace_id).hex, "<trace>")) == snapshot("""\
╭────────────────────────────────────────╮
│ Trace <trace> │
╰────────────────────────────────────────╯
Source: local (spans/<trace>.jsonl)

──────────────────────────────────────── Turn 1 — agent ────────────────────────────────────────
  → write(filePath='REPORT.md', content='PASS')
  write →
    wrote 4 bytes to REPORT.md
──────────────────────────────────────── Turn 2 — agent ────────────────────────────────────────
Wrote it.

View: https://hud.example/trace/<uuid>
""")
    assert "Skipped 1 incomplete span record" in result.stderr
