# ruff: noqa: E501 -- snapshots quote product messages verbatim
"""The Claude and Codex CLI agents run their CLI in the workspace over SSH.

A stub ``claude`` or ``codex`` on the workspace's PATH records what the agent
handed it (argv, the variables it set, stdin, the ``.hud_*`` files beside it),
prints a canned JSONL stream and exits as told. Rows pin that invocation, the
steps and trace the stream turned into, and how each exit ends the run.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import anyio
import pytest
from dirty_equals import IsStr
from inline_snapshot import snapshot

from hud.agents import ClaudeCLIAgent, CodexCLIAgent
from hud.agents.types import ClaudeCLIConfig, CodexCLIConfig
from hud.capabilities import Capability
from tests.agents.cli_stub import Stub, jsonl, process_exited
from tests.agents.support import run_task, workspace_env
from tests.harness import fake_screen, steps

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from hud import Environment
    from hud.agents.base import Agent
    from hud.eval import Run
    from tests.harness import HudEnv

TIMESTAMP = IsStr(regex=r"\d{4}-\d{2}-\d{2}T.*")

# ─── canned Claude CLI streams ──────────────────────────────────────────


def assistant(*content: dict[str, Any], stop: str = "end_turn") -> dict[str, Any]:
    return {
        "type": "assistant",
        "message": {
            "id": "msg",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "content": list(content),
            "stop_reason": stop,
            "stop_sequence": None,
            "usage": {"input_tokens": 3, "output_tokens": 2},
        },
    }


def tool_result(tool_use_id: str, *content: Any, is_error: bool = False) -> dict[str, Any]:
    return {
        "type": "user",
        "message": {
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": list(content),
                    "is_error": is_error,
                }
            ]
        },
    }


def result(text: str = "done", *, is_error: bool = False) -> dict[str, Any]:
    return {
        "type": "result",
        "subtype": "success",
        "is_error": is_error,
        "result": text,
        "session_id": "session-1",
        "duration_ms": 5,
        "num_turns": 2,
        "total_cost_usd": 0.01,
    }


WRITE = assistant(
    {"type": "text", "text": "editing"},
    {"type": "tool_use", "id": "tool-1", "name": "Write", "input": {"file_path": "a.txt"}},
    stop="tool_use",
)
WROTE = tool_result(
    "tool-1",
    {"type": "text", "text": "wrote a.txt"},
    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aW1n"}},
)
DONE = assistant({"type": "text", "text": "done"})
CLAUDE_STREAM = jsonl(WRITE, WROTE, DONE, result())


def claude_stub(root: Path, **behavior: Any) -> Stub:
    return Stub(root / "stub", "claude", **{"stdout": CLAUDE_STREAM, **behavior})


def codex_stub(root: Path, **behavior: Any) -> Stub:
    return Stub(root / "stub", "codex", **{"stdout": CODEX_STREAM, **behavior})


def cli_env(root: Path, stub: Stub, **options: Any) -> Environment:
    stub.install()
    return workspace_env(root / "ws", env=stub.shell_env(), **options)


def step_payloads(run: Run) -> list[dict[str, Any]]:
    """The agent and tool steps the run exported, timestamps checked and dropped."""
    found = []
    for payload in steps(run.trace_id):
        if payload["source"] in {"task", "user"}:
            continue
        for key in ("started_at", "ended_at"):
            if key in payload:
                assert payload.pop(key) == TIMESTAMP
        found.append(payload)
    return found


def invocation(capture: dict[str, Any], run: Run) -> dict[str, Any]:
    """The arguments (joined) and variables the agent launched its CLI with, the run's trace id as
    ``<trace>`` and a temporary ``CODEX_HOME`` as ``<temporary>``."""
    trace = str(run.trace_id)
    traces = (trace, uuid.UUID(trace).hex)

    def scrub(value: str) -> str:
        for form in traces:
            value = value.replace(form, "<trace>")
        return value

    env = {key: scrub(value) for key, value in capture["env"].items()}
    if "CODEX_HOME" in env:
        assert env["CODEX_HOME"] == IsStr(regex=r".*/hud-codex\.\w+")
        env["CODEX_HOME"] = "<temporary>"
    return {"argv": " ".join(scrub(arg) for arg in capture["argv"]), "env": env}


def outcome(run: Run) -> dict[str, Any]:
    return {
        "status": run.trace.status,
        "content": run.trace.content,
        "error": run.trace.error,
        "extra": run.trace.extra,
    }


# ─── Claude CLI ─────────────────────────────────────────────────────────


async def test_the_claude_cli_streams_its_turns_into_steps(hud_env: HudEnv, tmp_path: Path) -> None:
    hud_env.set(ANTHROPIC_API_KEY="anthropic-key")
    stub = claude_stub(tmp_path)

    run = await run_task(cli_env(tmp_path, stub), ClaudeCLIAgent(), prompt="build it")

    (capture,) = stub.captures()
    assert outcome(run) == snapshot(
        {
            "status": "completed",
            "content": "done",
            "error": None,
            "extra": {
                "subtype": "success",
                "session_id": "session-1",
                "duration_ms": 5,
                "num_turns": 2,
                "total_cost_usd": 0.01,
            },
        }
    )
    assert step_payloads(run) == snapshot(
        [
            {
                "step_id": 3,
                "source": "agent",
                "messages": [],
                "extra": {},
                "content": "editing",
                "tool_calls": [
                    {
                        "meta": {"citations_enabled": False},
                        "name": "Write",
                        "arguments": {"file_path": "a.txt"},
                        "id": "tool-1",
                    }
                ],
                "done": False,
                "finish_reason": "tool_use",
                "citations": [],
                "model": "claude-test",
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
            {
                "step_id": 4,
                "source": "tool",
                "messages": [],
                "extra": {},
                "call": {
                    "meta": {"citations_enabled": False},
                    "name": "Write",
                    "arguments": {"file_path": "a.txt"},
                    "id": "tool-1",
                },
                "result": {
                    "content": [
                        {"type": "text", "text": "wrote a.txt"},
                        {"type": "image", "data": "aW1n", "mimeType": "image/png"},
                    ],
                    "isError": False,
                    "call_id": "tool-1",
                },
            },
            {
                "step_id": 5,
                "source": "agent",
                "messages": [],
                "extra": {},
                "content": "done",
                "tool_calls": [],
                "done": True,
                "finish_reason": "end_turn",
                "citations": [],
                "model": "claude-test",
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
        ]
    )
    assert (capture["stdin"], capture["files"]) == snapshot(
        (
            '{"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": "build it"}]}}\n',
            {},
        )
    )


CLAUDE_INVOCATIONS = {
    "provider-key": (
        {"ANTHROPIC_API_KEY": "anthropic-key"},
        ClaudeCLIConfig(),
        snapshot(
            {
                "argv": "--verbose --input-format=stream-json --output-format=stream-json --print --permission-mode=bypassPermissions --allowedTools Read --allowedTools Write --allowedTools Edit --allowedTools Bash --allowedTools Glob --allowedTools Grep --allowedTools WebSearch --allowedTools WebFetch",
                "env": {
                    "IS_SANDBOX": "1",
                    "ANTHROPIC_MODEL": "claude-sonnet-5",
                    "ANTHROPIC_API_KEY": "anthropic-key",
                    "DISABLE_AUTOUPDATER": "1",
                    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                    "ANTHROPIC_SMALL_FAST_MODEL": "claude-sonnet-5",
                },
            }
        ),
    ),
    "gateway-without-a-provider-key": (
        {"HUD_API_KEY": "hud-key"},
        ClaudeCLIConfig(),
        snapshot(
            {
                "argv": "--verbose --input-format=stream-json --output-format=stream-json --print --permission-mode=bypassPermissions --allowedTools Read --allowedTools Write --allowedTools Edit --allowedTools Bash --allowedTools Glob --allowedTools Grep --allowedTools WebSearch --allowedTools WebFetch",
                "env": {
                    "ANTHROPIC_CUSTOM_HEADERS": "Trace-Id: <trace>",
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "claude-sonnet-5",
                    "IS_SANDBOX": "1",
                    "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
                    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                    "CLAUDE_CODE_SUBAGENT_MODEL": "claude-sonnet-5",
                    "ANTHROPIC_BASE_URL": "http://127.0.0.1:9/gateway",
                    "DISABLE_AUTOUPDATER": "1",
                    "ANTHROPIC_DEFAULT_OPUS_MODEL": "claude-sonnet-5",
                    "ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-sonnet-5",
                    "ANTHROPIC_MODEL": "claude-sonnet-5",
                    "ANTHROPIC_SMALL_FAST_MODEL": "claude-sonnet-5",
                    "ANTHROPIC_API_KEY": "hud-key",
                },
            }
        ),
    ),
    "gateway-forced-over-a-provider-key": (
        {"HUD_API_KEY": "hud-key", "ANTHROPIC_API_KEY": "anthropic-key"},
        ClaudeCLIConfig(gateway=True),
        snapshot(
            {
                "argv": "--verbose --input-format=stream-json --output-format=stream-json --print --permission-mode=bypassPermissions --allowedTools Read --allowedTools Write --allowedTools Edit --allowedTools Bash --allowedTools Glob --allowedTools Grep --allowedTools WebSearch --allowedTools WebFetch",
                "env": {
                    "ANTHROPIC_CUSTOM_HEADERS": "Trace-Id: <trace>",
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "claude-sonnet-5",
                    "IS_SANDBOX": "1",
                    "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
                    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                    "CLAUDE_CODE_SUBAGENT_MODEL": "claude-sonnet-5",
                    "ANTHROPIC_BASE_URL": "http://127.0.0.1:9/gateway",
                    "DISABLE_AUTOUPDATER": "1",
                    "ANTHROPIC_DEFAULT_OPUS_MODEL": "claude-sonnet-5",
                    "ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-sonnet-5",
                    "ANTHROPIC_MODEL": "claude-sonnet-5",
                    "ANTHROPIC_SMALL_FAST_MODEL": "claude-sonnet-5",
                    "ANTHROPIC_API_KEY": "hud-key",
                },
            }
        ),
    ),
    "configured-options": (
        {"ANTHROPIC_API_KEY": "anthropic-key"},
        ClaudeCLIConfig(
            model="claude-opus-4-8",
            reasoning_effort="max",
            max_steps=3,
            system_prompt="Be brief.",
            allowed_tools=["Read", "Bash"],
            permission_mode="acceptEdits",
        ),
        snapshot(
            {
                "argv": "--verbose --input-format=stream-json --output-format=stream-json --print --permission-mode=acceptEdits --max-turns=3 --effort max --system-prompt Be brief. --allowedTools Read --allowedTools Bash",
                "env": {
                    "IS_SANDBOX": "1",
                    "ANTHROPIC_MODEL": "claude-opus-4-8",
                    "ANTHROPIC_API_KEY": "anthropic-key",
                    "DISABLE_AUTOUPDATER": "1",
                    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                    "ANTHROPIC_SMALL_FAST_MODEL": "claude-opus-4-8",
                },
            }
        ),
    ),
}


@pytest.mark.parametrize(
    ("variables", "config", "expected"),
    CLAUDE_INVOCATIONS.values(),
    ids=CLAUDE_INVOCATIONS.keys(),
)
async def test_the_claude_cli_is_launched_with_the_routing_and_options_configured(
    variables: dict[str, str],
    config: ClaudeCLIConfig,
    expected: Any,
    hud_env: HudEnv,
    tmp_path: Path,
) -> None:
    hud_env.set(**variables)
    stub = claude_stub(tmp_path)

    run = await run_task(cli_env(tmp_path, stub), ClaudeCLIAgent(config))

    (capture,) = stub.captures()
    assert invocation(capture, run) == expected
    assert run.trace.status == "completed"


CLAUDE_EXITS = {
    "nonzero-exit-without-output": (
        {"stdout": "", "stderr": "boom", "exit": 1},
        snapshot(
            {
                "status": "error",
                "content": None,
                "error": "[agent loop] RuntimeError: boom",
                "extra": {"returncode": 1, "stderr": "boom"},
            }
        ),
    ),
    "nonzero-exit-after-a-result-with-stderr": (
        {"stderr": "transport failed", "exit": 1},
        snapshot(
            {
                "status": "error",
                "content": "done",
                "error": "[agent loop] RuntimeError: transport failed",
                "extra": {
                    "subtype": "success",
                    "session_id": "session-1",
                    "duration_ms": 5,
                    "num_turns": 2,
                    "total_cost_usd": 0.01,
                    "returncode": 1,
                    "stderr": "transport failed",
                },
            }
        ),
    ),
    "nonzero-exit-after-a-result-without-stderr": (
        {"exit": 1},
        snapshot(
            {
                "status": "completed",
                "content": "done",
                "error": None,
                "extra": {
                    "subtype": "success",
                    "session_id": "session-1",
                    "duration_ms": 5,
                    "num_turns": 2,
                    "total_cost_usd": 0.01,
                    "returncode": 1,
                },
            }
        ),
    ),
    "zero-exit-without-a-result": (
        {"stdout": jsonl(WRITE, WROTE, DONE)},
        snapshot(
            {
                "status": "error",
                "content": "done",
                "error": "[agent loop] RuntimeError: claude CLI exited without a result event",
                "extra": {},
            }
        ),
    ),
    "result-reports-an-error": (
        {"stdout": jsonl(DONE, result("rate limited upstream", is_error=True))},
        snapshot(
            {
                "status": "error",
                "content": "rate limited upstream",
                "error": "[agent loop] RuntimeError: rate limited upstream",
                "extra": {
                    "subtype": "success",
                    "session_id": "session-1",
                    "duration_ms": 5,
                    "num_turns": 2,
                    "total_cost_usd": 0.01,
                },
            }
        ),
    ),
    "tool-call-left-without-a-result": (
        {"stdout": jsonl(WRITE, result())},
        snapshot(
            {
                "status": "error",
                "content": "done",
                "error": "[agent loop] RuntimeError: claude CLI exited without results for tool calls: tool-1",
                "extra": {
                    "subtype": "success",
                    "session_id": "session-1",
                    "duration_ms": 5,
                    "num_turns": 2,
                    "total_cost_usd": 0.01,
                },
            }
        ),
    ),
}


@pytest.mark.parametrize(("behavior", "expected"), CLAUDE_EXITS.values(), ids=CLAUDE_EXITS.keys())
async def test_how_the_claude_cli_exits_decides_how_the_run_ends(
    behavior: dict[str, Any], expected: Any, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(ANTHROPIC_API_KEY="anthropic-key")
    stub = claude_stub(tmp_path, **behavior)

    run = await run_task(cli_env(tmp_path, stub), ClaudeCLIAgent())

    assert outcome(run) == expected


async def test_claude_cli_tool_results_keep_documents_and_name_unknown_blocks(
    hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(ANTHROPIC_API_KEY="anthropic-key")
    read = assistant(
        {"type": "tool_use", "id": "tool-1", "name": "Read", "input": {}}, stop="tool_use"
    )
    pdf = {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0="},
    }
    stub = claude_stub(
        tmp_path,
        stdout=jsonl(read, tool_result("tool-1", pdf, {"type": "novelty"}), DONE, result()),
    )

    run = await run_task(cli_env(tmp_path, stub), ClaudeCLIAgent())

    tool = next(payload for payload in step_payloads(run) if payload["source"] == "tool")
    assert (run.trace.status, tool["result"]["content"]) == snapshot(
        (
            "completed",
            [
                {
                    "type": "resource",
                    "resource": {
                        "uri": "document://tool-1/0",
                        "mimeType": "application/pdf",
                        "blob": "JVBERi0=",
                    },
                },
                {"type": "text", "text": "[unsupported novelty block]"},
            ],
        )
    )


async def test_mcp_servers_reach_the_claude_cli_in_a_config_removed_after_the_run(
    hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(ANTHROPIC_API_KEY="anthropic-key")
    stub = claude_stub(tmp_path)
    servers = (
        Capability.mcp(name="database", url="http://db.internal:8000/mcp", auth_token="secret"),
        Capability.mcp(name="search", url="http://search.internal:8000/sse", transport="sse"),
    )

    await run_task(cli_env(tmp_path, stub, capabilities=servers), ClaudeCLIAgent())

    (capture,) = stub.captures()
    assert capture["argv"][-2:] == ["--mcp-config", ".hud_mcp_config.json"]
    assert capture["files"] == snapshot(
        {
            ".hud_mcp_config.json": """\
{
  "mcpServers": {
    "database": {
      "type": "http",
      "url": "http://db.internal:8000/mcp",
      "headers": {
        "Authorization": "Bearer secret"
      }
    },
    "search": {
      "type": "sse",
      "url": "http://search.internal:8000/sse"
    }
  }
}\
"""
        }
    )
    assert sorted(path.name for path in (tmp_path / "ws").iterdir()) == []


async def test_the_claude_cli_drives_the_screen_through_the_computer_bridge(
    hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(ANTHROPIC_API_KEY="anthropic-key")
    stub = claude_stub(
        tmp_path,
        computer=[
            {"action": "left_click", "coordinate": [3, 4]},
            {"action": "frobnicate"},
        ],
    )

    async with fake_screen() as screen:
        env = cli_env(tmp_path, stub, capabilities=(Capability.rfb(url=screen.url),))
        run = await run_task(env, ClaudeCLIAgent())

    (capture,) = stub.captures()
    assert run.trace.status == "completed"
    assert capture["computer"] == snapshot(
        [
            {"isError": False, "content": ["image"]},
            {"isError": True, "content": ["unsupported computer action: 'frobnicate'"]},
        ]
    )
    assert [(event.x, event.y, event.buttons) for event in screen.pointer_events()] == [
        (3, 4, 0),
        (3, 4, 1),
        (3, 4, 0),
    ]


async def test_an_mcp_server_named_like_the_computer_bridge_is_refused(
    hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(ANTHROPIC_API_KEY="anthropic-key")
    stub = claude_stub(tmp_path)

    async with fake_screen() as screen:
        clash = (
            Capability.rfb(url=screen.url),
            Capability.mcp(name="computer-use", url="http://tools.internal:8000/mcp"),
        )
        run = await run_task(cli_env(tmp_path, stub, capabilities=clash), ClaudeCLIAgent())

    assert (run.trace.error, stub.captures()) == snapshot(
        ("[agent loop] RuntimeError: duplicate MCP server name 'computer-use'", [])
    )


async def test_the_claude_cli_needs_a_credential_to_route_through_the_gateway(
    tmp_path: Path,
) -> None:
    stub = claude_stub(tmp_path)

    run = await run_task(cli_env(tmp_path, stub), ClaudeCLIAgent(ClaudeCLIConfig(gateway=True)))

    assert (run.trace.error, stub.captures()) == snapshot(
        ("[agent loop] ValueError: HUD_API_KEY is required for HUD gateway routing", [])
    )


# ─── both CLIs ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Cli:
    stub: Callable[..., Stub]
    agent: Callable[[], Agent]
    variables: dict[str, str]


CLIS = {
    "claude": Cli(claude_stub, ClaudeCLIAgent, {"ANTHROPIC_API_KEY": "anthropic-key"}),
    "codex": Cli(
        lambda root, **kw: codex_stub(root, **kw), CodexCLIAgent, {"OPENAI_API_KEY": "openai-key"}
    ),
}


@pytest.mark.parametrize("cli", CLIS.values(), ids=CLIS.keys())
async def test_a_missing_cli_names_the_runtime_platform(
    cli: Cli, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(**cli.variables)
    stub = cli.stub(tmp_path)
    stub.install()
    env = workspace_env(tmp_path / "ws", env=stub.shell_env(path="/usr/bin:/bin"))

    run = await run_task(env, cli.agent())

    assert run.trace.error == IsStr(
        regex=r".* is unavailable for runtime platform linux-\w+; install it in the "
        r"environment or provide a managed runtime bundle"
    )


@pytest.mark.parametrize("cli", CLIS.values(), ids=CLIS.keys())
async def test_one_cli_agent_drives_concurrent_rollouts(
    cli: Cli, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(**cli.variables)
    stub = cli.stub(tmp_path)
    env = cli_env(tmp_path, stub)
    agent = cli.agent()

    runs = await asyncio.gather(
        run_task(env, agent, prompt="first"), run_task(env, agent, prompt="second")
    )

    stdins = sorted(capture["stdin"] for capture in stub.captures())
    assert [run.trace.status for run in runs] == ["completed", "completed"]
    assert [("first" in stdin, "second" in stdin) for stdin in stdins] == [
        (True, False),
        (False, True),
    ]


@pytest.mark.parametrize("cli", CLIS.values(), ids=CLIS.keys())
async def test_an_agent_timeout_stops_the_cli(cli: Cli, hud_env: HudEnv, tmp_path: Path) -> None:
    hud_env.set(**cli.variables)
    first_line = (CLAUDE_STREAM if cli.agent is ClaudeCLIAgent else CODEX_STREAM).splitlines()[0]
    stub = cli.stub(tmp_path, stdout=first_line + "\n", hang=True)
    agent = type(cli.agent()).load({**cli.agent().dump(), "timeout_seconds": 2.0})

    run = await run_task(cli_env(tmp_path, stub), agent)

    (capture,) = stub.captures()
    assert (run.trace.status, run.trace.stop_reason) == ("error", "timeout")
    assert process_exited(capture["pid"])


# ─── Codex CLI ──────────────────────────────────────────────────────────


def item(kind: str, item_id: str, **fields: Any) -> dict[str, Any]:
    return {"type": "item.completed", "item": {"id": item_id, "type": kind, **fields}}


CODEX_ITEMS = [
    {"type": "thread.started", "thread_id": "thread-1"},
    {"type": "turn.started"},
    {
        "type": "item.started",
        "item": {
            "id": "cmd-1",
            "type": "command_execution",
            "command": "pytest -q",
            "aggregated_output": "",
            "exit_code": None,
            "status": "in_progress",
        },
    },
    item(
        "command_execution",
        "cmd-1",
        command="pytest -q",
        aggregated_output="1 passed\n",
        exit_code=0,
        status="completed",
    ),
    item(
        "command_execution",
        "cmd-2",
        command="false",
        aggregated_output="",
        exit_code=1,
        status="completed",
    ),
    item(
        "file_change",
        "patch-1",
        changes=[{"path": "calc.py", "kind": "update"}],
        status="completed",
    ),
    item(
        "mcp_tool_call",
        "mcp-1",
        server="db",
        tool="query",
        arguments={"sql": "select 42"},
        result={"content": [{"type": "text", "text": "42"}], "structured_content": {"answer": 42}},
        error=None,
        status="completed",
    ),
    item(
        "mcp_tool_call",
        "mcp-2",
        server="db",
        tool="query",
        arguments={"sql": "drop"},
        result=None,
        error={"message": "denied"},
        status="failed",
    ),
    item("web_search", "search-1", query="HUD evals", action={"type": "search"}),
    item("todo_list", "todo-1", items=[{"text": "ship", "completed": False}]),
    item("reasoning", "reason-1", text="The test now passes."),
    item("agent_message", "message-1", text="Implemented and verified."),
]
CODEX_DONE = {
    "type": "turn.completed",
    "usage": {"input_tokens": 20, "cached_input_tokens": 5, "output_tokens": 8},
}
CODEX_STREAM = jsonl(*CODEX_ITEMS, CODEX_DONE)


async def test_codex_events_become_steps(hud_env: HudEnv, tmp_path: Path) -> None:
    hud_env.set(OPENAI_API_KEY="openai-key")
    stub = codex_stub(tmp_path)

    run = await run_task(cli_env(tmp_path, stub), CodexCLIAgent(), prompt="Fix the failing test")

    (capture,) = stub.captures()
    assert capture["stdin"] == "Fix the failing test"
    assert not await anyio.Path(capture["env"]["CODEX_HOME"]).exists()
    assert outcome(run) == snapshot(
        {
            "status": "completed",
            "content": "Implemented and verified.",
            "error": None,
            "extra": {
                "codex_thread_id": "thread-1",
                "usage": {"input_tokens": 20, "cached_input_tokens": 5, "output_tokens": 8},
            },
        }
    )
    assert step_payloads(run) == snapshot(
        [
            {
                "step_id": 3,
                "source": "tool",
                "messages": [],
                "extra": {
                    "codex_item": {
                        "id": "cmd-1",
                        "type": "command_execution",
                        "command": "pytest -q",
                        "aggregated_output": "1 passed\n",
                        "exit_code": 0,
                        "status": "completed",
                    }
                },
                "call": {"name": "shell", "arguments": {"command": "pytest -q"}, "id": "cmd-1"},
                "result": {
                    "content": [{"type": "text", "text": "1 passed\n"}],
                    "isError": False,
                    "call_id": "cmd-1",
                },
            },
            {
                "step_id": 4,
                "source": "tool",
                "messages": [],
                "extra": {
                    "codex_item": {
                        "id": "cmd-2",
                        "type": "command_execution",
                        "command": "false",
                        "aggregated_output": "",
                        "exit_code": 1,
                        "status": "completed",
                    }
                },
                "call": {"name": "shell", "arguments": {"command": "false"}, "id": "cmd-2"},
                "result": {
                    "content": [{"type": "text", "text": ""}],
                    "isError": True,
                    "call_id": "cmd-2",
                },
            },
            {
                "step_id": 5,
                "source": "tool",
                "messages": [],
                "extra": {
                    "codex_item": {
                        "id": "patch-1",
                        "type": "file_change",
                        "changes": [{"path": "calc.py", "kind": "update"}],
                        "status": "completed",
                    }
                },
                "call": {
                    "name": "apply_patch",
                    "arguments": {"changes": [{"path": "calc.py", "kind": "update"}]},
                    "id": "patch-1",
                },
                "result": {
                    "content": [{"type": "text", "text": "update: calc.py"}],
                    "isError": False,
                    "call_id": "patch-1",
                },
            },
            {
                "step_id": 6,
                "source": "tool",
                "messages": [],
                "extra": {
                    "codex_item": {
                        "id": "mcp-1",
                        "type": "mcp_tool_call",
                        "server": "db",
                        "tool": "query",
                        "arguments": {"sql": "select 42"},
                        "result": {
                            "content": [{"type": "text", "text": "42"}],
                            "structured_content": {"answer": 42},
                        },
                        "error": None,
                        "status": "completed",
                    }
                },
                "call": {
                    "name": "query",
                    "arguments": {"sql": "select 42"},
                    "id": "mcp-1",
                    "provider_name": "db.query",
                },
                "result": {
                    "content": [{"type": "text", "text": "42"}],
                    "structuredContent": {"answer": 42},
                    "isError": False,
                    "call_id": "mcp-1",
                },
            },
            {
                "step_id": 7,
                "source": "tool",
                "messages": [],
                "extra": {
                    "codex_item": {
                        "id": "mcp-2",
                        "type": "mcp_tool_call",
                        "server": "db",
                        "tool": "query",
                        "arguments": {"sql": "drop"},
                        "result": None,
                        "error": {"message": "denied"},
                        "status": "failed",
                    }
                },
                "call": {
                    "name": "query",
                    "arguments": {"sql": "drop"},
                    "id": "mcp-2",
                    "provider_name": "db.query",
                },
                "result": {
                    "content": [{"type": "text", "text": "denied"}],
                    "isError": True,
                    "call_id": "mcp-2",
                },
            },
            {
                "step_id": 8,
                "source": "tool",
                "messages": [],
                "extra": {
                    "codex_item": {
                        "id": "search-1",
                        "type": "web_search",
                        "query": "HUD evals",
                        "action": {"type": "search"},
                    }
                },
                "call": {
                    "name": "web_search",
                    "arguments": {"query": "HUD evals", "action": {"type": "search"}},
                    "id": "search-1",
                },
                "result": {
                    "content": [{"type": "text", "text": '{"type":"search"}'}],
                    "isError": False,
                    "call_id": "search-1",
                },
            },
            {
                "step_id": 9,
                "source": "agent",
                "messages": [],
                "extra": {
                    "codex_item": {
                        "id": "todo-1",
                        "type": "todo_list",
                        "items": [{"text": "ship", "completed": False}],
                    }
                },
            },
            {
                "step_id": 10,
                "source": "agent",
                "messages": [],
                "extra": {},
                "reasoning": "The test now passes.",
                "tool_calls": [],
                "done": False,
                "citations": [],
                "raw": {"id": "reason-1", "type": "reasoning", "text": "The test now passes."},
                "model": "gpt-5.6-sol",
            },
            {
                "step_id": 11,
                "source": "agent",
                "messages": [],
                "extra": {},
                "content": "Implemented and verified.",
                "tool_calls": [],
                "done": False,
                "citations": [],
                "raw": {
                    "id": "message-1",
                    "type": "agent_message",
                    "text": "Implemented and verified.",
                },
                "model": "gpt-5.6-sol",
            },
        ]
    )


CODEX_INVOCATIONS = {
    "provider-key": (
        {"OPENAI_API_KEY": "openai-key"},
        CodexCLIConfig(),
        snapshot(
            {
                "argv": "exec --json --ephemeral --skip-git-repo-check --color never --sandbox workspace-write --model gpt-5.6-sol -",
                "env": {"CODEX_API_KEY": "openai-key", "CODEX_HOME": "<temporary>"},
            }
        ),
    ),
    "gateway-without-a-provider-key": (
        {"HUD_API_KEY": "hud-key"},
        CodexCLIConfig(),
        snapshot(
            {
                "argv": 'exec --json --ephemeral --skip-git-repo-check --color never --sandbox workspace-write --model gpt-5.6-sol -c model_provider="hud" -c model_providers.hud.name="HUD" -c model_providers.hud.base_url="http://127.0.0.1:9/gateway" -c model_providers.hud.env_key="HUD_API_KEY" -c model_providers.hud.wire_api="responses" -c model_providers.hud.http_headers={"Trace-Id"="<trace>"} -',
                "env": {"HUD_API_KEY": "hud-key", "CODEX_HOME": "<temporary>"},
            }
        ),
    ),
    "gateway-forced-over-a-provider-key": (
        {"HUD_API_KEY": "hud-key", "OPENAI_API_KEY": "openai-key"},
        CodexCLIConfig(gateway=True),
        snapshot(
            {
                "argv": 'exec --json --ephemeral --skip-git-repo-check --color never --sandbox workspace-write --model gpt-5.6-sol -c model_provider="hud" -c model_providers.hud.name="HUD" -c model_providers.hud.base_url="http://127.0.0.1:9/gateway" -c model_providers.hud.env_key="HUD_API_KEY" -c model_providers.hud.wire_api="responses" -c model_providers.hud.http_headers={"Trace-Id"="<trace>"} -',
                "env": {"HUD_API_KEY": "hud-key", "CODEX_HOME": "<temporary>"},
            }
        ),
    ),
    "ambient-login": (
        {},
        CodexCLIConfig(),
        snapshot(
            {
                "argv": "exec --json --ephemeral --skip-git-repo-check --color never --sandbox workspace-write --model gpt-5.6-sol -",
                "env": {},
            }
        ),
    ),
    "configured-options": (
        {"OPENAI_API_KEY": "openai-key"},
        CodexCLIConfig(model="gpt-5.6", sandbox="read-only", reasoning_effort="max"),
        snapshot(
            {
                "argv": 'exec --json --ephemeral --skip-git-repo-check --color never --sandbox read-only --model gpt-5.6 -c model_reasoning_effort="max" -',
                "env": {"CODEX_API_KEY": "openai-key", "CODEX_HOME": "<temporary>"},
            }
        ),
    ),
}


@pytest.mark.parametrize(
    ("variables", "config", "expected"),
    CODEX_INVOCATIONS.values(),
    ids=CODEX_INVOCATIONS.keys(),
)
async def test_codex_is_launched_with_the_routing_and_options_configured(
    variables: dict[str, str],
    config: CodexCLIConfig,
    expected: Any,
    hud_env: HudEnv,
    tmp_path: Path,
) -> None:
    hud_env.set(**variables)
    stub = codex_stub(tmp_path)

    run = await run_task(cli_env(tmp_path, stub), CodexCLIAgent(config))

    (capture,) = stub.captures()
    assert invocation(capture, run) == expected


CODEX_EXITS = {
    "turn-failed": (
        {
            "stdout": jsonl(
                {"type": "turn.started"},
                {"type": "turn.failed", "error": {"message": "model unavailable"}},
            )
        },
        snapshot(
            {
                "status": "error",
                "content": None,
                "error": "[agent loop] RuntimeError: model unavailable",
                "extra": {},
            }
        ),
    ),
    "nonzero-exit": (
        {"stdout": "", "stderr": "authentication failed", "exit": 1},
        snapshot(
            {
                "status": "error",
                "content": None,
                "error": "[agent loop] RuntimeError: authentication failed",
                "extra": {"returncode": 1, "stderr": "authentication failed"},
            }
        ),
    ),
    "nonzero-exit-with-a-structured-error": (
        {
            "stdout": jsonl({"type": "error", "message": "gateway rejected streaming"}),
            "stderr": "noisy",
            "exit": 1,
        },
        snapshot(
            {
                "status": "error",
                "content": None,
                "error": "[agent loop] RuntimeError: gateway rejected streaming",
                "extra": {"returncode": 1},
            }
        ),
    ),
    "zero-exit-without-turn-completed": (
        {"stdout": jsonl(*CODEX_ITEMS)},
        snapshot(
            {
                "status": "error",
                "content": "Implemented and verified.",
                "error": "[agent loop] RuntimeError: codex CLI exited without a turn.completed event",
                "extra": {"codex_thread_id": "thread-1"},
            }
        ),
    ),
}


@pytest.mark.parametrize(("behavior", "expected"), CODEX_EXITS.values(), ids=CODEX_EXITS.keys())
async def test_how_codex_exits_decides_how_the_run_ends(
    behavior: dict[str, Any], expected: Any, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(OPENAI_API_KEY="openai-key")
    stub = codex_stub(tmp_path, **behavior)

    run = await run_task(cli_env(tmp_path, stub), CodexCLIAgent())

    assert outcome(run) == expected
