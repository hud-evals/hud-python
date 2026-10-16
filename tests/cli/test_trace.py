"""``hud trace``: a rollout's turns, from local spans or the platform."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import pytest

from hud import Environment
from hud.agents.openai_compatible import OpenAIChatAgent
from hud.agents.types import OpenAIChatConfig
from hud.eval import LocalRuntime, Task, rollout
from tests.harness import call, say

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


def test_a_platform_trace_prints_the_models_text_verbatim(hud: Hud, events: FakeServices) -> None:
    result = hud("trace", COMPACT_TRACE_ID)

    assert result.exit_code == 0, result
    for shown in (EVENTS[0]["text"], EVENTS[0]["reasoning"], "exit status 1", "max steps reached"):
        assert shown in result.stdout


def test_trace_json_is_the_event_list(hud: Hud, events: FakeServices) -> None:
    result = hud("trace", "get", TRACE_ID, "--json")

    assert result.exit_code == 0, result
    assert result.json == EVENTS
    (request,) = events.requests("api", "GET", "/v2/trace/{id}/events")
    assert request.params == {"id": TRACE_ID}


@pytest.mark.parametrize(
    ("argv", "exit_code", "error"),
    [
        pytest.param([TRACE_ID], 1, "not_found", id="missing"),
        pytest.param(["unknown"], 2, "usage", id="malformed"),
    ],
)
def test_a_missing_or_malformed_trace_is_an_error_document(
    hud: Hud, platform: FakeServices, argv: list[str], exit_code: int, error: str
) -> None:
    platform.route("api", "GET", "/v2/trace/{id}/events", status=404, json={"detail": "missing"})

    result = hud("trace", *argv, "--json")

    assert (result.exit_code, result.json["error"]) == (exit_code, error), result


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
    assert [(event["kind"], event.get("tool_name")) for event in result.json] == [
        ("agent_message", None),
        ("tool_call", "write"),
        ("agent_message", None),
    ]
    assert result.json[1]["arguments"] == {"filePath": "REPORT.md", "content": "PASS"}
    assert "4 bytes" in result.json[1]["result_text"]
    assert result.json[2]["text"] == "Wrote it."


async def test_a_local_trace_renders_and_skips_a_record_cut_short(
    hud: Hud, models: Models, hud_env: HudEnv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace_id = await write_a_report(models, hud_env, tmp_path, monkeypatch)
    spans = tmp_path / "spans" / f"{uuid.UUID(trace_id).hex}.jsonl"
    complete = spans.read_text()
    spans.write_text(complete + complete.splitlines()[-1][:40])

    result = hud("trace", trace_id, cwd=tmp_path)

    assert result.exit_code == 0, result
    assert "Wrote it." in result.stdout
    assert "Skipped 1 incomplete span record" in result.stderr


def test_a_trace_missing_from_the_local_spans_is_read_from_the_platform(
    hud: Hud, events: FakeServices, hud_env: HudEnv, tmp_path: Path
) -> None:
    spans = tmp_path / "spans"
    spans.mkdir()
    (spans / f"{uuid.uuid4().hex}.jsonl").write_text("")
    hud_env.set(HUD_TELEMETRY_LOCAL_DIR=str(spans))

    result = hud("trace", "get", TRACE_ID, "--json")

    assert result.exit_code == 0, result
    assert result.json == EVENTS
    (request,) = events.requests("api", "GET", "/v2/trace/{id}/events")
    assert request.params == {"id": TRACE_ID}
