"""The Claude and Codex CLI agents run their CLI in the workspace over SSH.

A stub ``claude`` or ``codex`` on the workspace's PATH records what the agent
handed it (argv, the variables it set, stdin, the ``.hud_*`` files beside it),
prints a canned JSONL stream and exits as told. Rows pin that invocation, the
steps and trace the stream turned into, and how each exit ends the run.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest
from dirty_equals import IsStr

from hud.agents import ClaudeCLIAgent, CodexCLIAgent
from hud.agents.types import ClaudeCLIConfig, CodexCLIConfig
from hud.capabilities import Capability, Connection
from hud.environment.egress import ANY_HOST
from tests.agents.cli_stub import Stub, jsonl, process_exited
from tests.agents.support import run_task, workspace_env
from tests.harness import fake_screen

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from hud import Environment
    from hud.agents.base import Agent
    from hud.eval import Run
    from tests.harness import HudEnv


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
DONE = assistant({"type": "text", "text": "done"})
CLAUDE_STREAM = jsonl(DONE, result())


def claude_stub(root: Path, **behavior: Any) -> Stub:
    behavior.setdefault("stdout", CLAUDE_STREAM)
    return Stub(root / "stub", "claude", **behavior)


def codex_stub(root: Path, **behavior: Any) -> Stub:
    behavior.setdefault("stdout", CODEX_STREAM)
    return Stub(root / "stub", "codex", **behavior)


def cli_env(root: Path, stub: Stub, **options: Any) -> Environment:
    stub.install()
    return workspace_env(root / "ws", env=stub.shell_env(), **options)


def launched(capture: dict[str, Any], run: Run, variables: tuple[str, ...]) -> dict[str, Any]:
    """The argv the agent launched its CLI with and the named variables it set, with
    the run's trace id as ``<trace>``."""
    trace = str(run.trace_id)

    def scrub(value: str) -> str:
        return value.replace(trace, "<trace>").replace(uuid.UUID(trace).hex, "<trace>")

    env = capture["env"]
    return {
        "argv": " ".join(scrub(arg) for arg in capture["argv"]),
        **{name: scrub(env[name]) for name in variables if name in env},
    }


def outcome(run: Run) -> tuple[str | None, str | None, str | None]:
    return (run.trace.status, run.trace.content, run.trace.error)


# ─── Claude CLI ─────────────────────────────────────────────────────────


CLAUDE_ARGV = (
    "--verbose --input-format=stream-json --output-format=stream-json --print "
    "--permission-mode=bypassPermissions "
    + " ".join(
        f"--allowedTools {tool}"
        for tool in ("Read", "Write", "Edit", "Bash", "Glob", "Grep", "WebSearch", "WebFetch")
    )
)
CLAUDE_ROUTING = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_CUSTOM_HEADERS",
    "ANTHROPIC_MODEL",
)
CLAUDE_GATEWAY = {
    "argv": CLAUDE_ARGV,
    "ANTHROPIC_API_KEY": "hud-key",
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:9/gateway",
    "ANTHROPIC_CUSTOM_HEADERS": "Trace-Id: <trace>",
    "ANTHROPIC_MODEL": "claude-sonnet-5",
}
CLAUDE_INVOCATIONS = {
    "provider-key": (
        {"ANTHROPIC_API_KEY": "anthropic-key"},
        ClaudeCLIConfig(),
        {
            "argv": CLAUDE_ARGV,
            "ANTHROPIC_API_KEY": "anthropic-key",
            "ANTHROPIC_MODEL": "claude-sonnet-5",
        },
    ),
    "gateway-without-a-provider-key": (
        {"HUD_API_KEY": "hud-key"},
        ClaudeCLIConfig(),
        CLAUDE_GATEWAY,
    ),
    "gateway-forced-over-a-provider-key": (
        {"HUD_API_KEY": "hud-key", "ANTHROPIC_API_KEY": "anthropic-key"},
        ClaudeCLIConfig(gateway=True),
        CLAUDE_GATEWAY,
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
        {
            "argv": "--verbose --input-format=stream-json --output-format=stream-json --print "
            "--permission-mode=acceptEdits --max-turns=3 --effort max --system-prompt Be brief. "
            "--allowedTools Read --allowedTools Bash",
            "ANTHROPIC_API_KEY": "anthropic-key",
            "ANTHROPIC_MODEL": "claude-opus-4-8",
        },
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
    assert launched(capture, run, CLAUDE_ROUTING) == expected
    assert run.trace.status == "completed"


CLAUDE_EXITS = {
    "nonzero-exit-without-output": (
        {"stdout": "", "stderr": "boom", "exit": 1},
        ("error", None, "[agent loop] RuntimeError: boom"),
    ),
    "nonzero-exit-after-a-result-with-stderr": (
        {"stderr": "transport failed", "exit": 1},
        ("error", "done", "[agent loop] RuntimeError: transport failed"),
    ),
    "nonzero-exit-after-a-result-without-stderr": ({"exit": 1}, ("completed", "done", None)),
    "zero-exit-without-a-result": (
        {"stdout": jsonl(DONE)},
        ("error", "done", "[agent loop] RuntimeError: claude CLI exited without a result event"),
    ),
    "result-reports-an-error": (
        {"stdout": jsonl(DONE, result("rate limited upstream", is_error=True))},
        ("error", "rate limited upstream", "[agent loop] RuntimeError: rate limited upstream"),
    ),
    "tool-call-left-without-a-result": (
        {"stdout": jsonl(WRITE, result())},
        (
            "error",
            "done",
            "[agent loop] RuntimeError: claude CLI exited without results for tool calls: tool-1",
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
    assert json.loads(capture["files"][".hud_mcp_config.json"]) == {
        "mcpServers": {
            "database": {
                "type": "http",
                "url": "http://db.internal:8000/mcp",
                "headers": {"Authorization": "Bearer secret"},
            },
            "search": {"type": "sse", "url": "http://search.internal:8000/sse"},
        }
    }
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
    assert capture["computer"] == [
        {"isError": False, "content": ["image"]},
        {"isError": True, "content": ["unsupported computer action: 'frobnicate'"]},
    ]
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

    assert (run.trace.error, stub.captures()) == (
        "[agent loop] RuntimeError: duplicate MCP server name 'computer-use'",
        [],
    )


async def test_the_claude_cli_needs_a_credential_to_route_through_the_gateway(
    tmp_path: Path,
) -> None:
    stub = claude_stub(tmp_path)

    run = await run_task(cli_env(tmp_path, stub), ClaudeCLIAgent(ClaudeCLIConfig(gateway=True)))

    assert (run.trace.error, stub.captures()) == (
        "[agent loop] ValueError: HUD_API_KEY is required for HUD gateway routing",
        [],
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
        regex=r".* is unavailable for runtime platform [a-z]+-\w+; install it in the "
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
    item("agent_message", "message-1", text="Implemented and verified."),
]
CODEX_DONE = {
    "type": "turn.completed",
    "usage": {"input_tokens": 20, "cached_input_tokens": 5, "output_tokens": 8},
}
CODEX_STREAM = jsonl(*CODEX_ITEMS, CODEX_DONE)


CODEX_ARGV = (
    "exec --json --ephemeral --skip-git-repo-check --color never --sandbox workspace-write "
    "--model gpt-5.6-sol"
)
CODEX_THROUGH_GATEWAY = {
    "argv": f"{CODEX_ARGV} "
    + " ".join(
        f"-c {option}"
        for option in (
            'model_provider="hud"',
            'model_providers.hud.name="HUD"',
            'model_providers.hud.base_url="http://127.0.0.1:9/gateway"',
            'model_providers.hud.env_key="HUD_API_KEY"',
            'model_providers.hud.wire_api="responses"',
            'model_providers.hud.http_headers={"Trace-Id"="<trace>"}',
        )
    )
    + " -",
    "HUD_API_KEY": "hud-key",
}
CODEX_ROUTING = ("CODEX_API_KEY", "HUD_API_KEY", "OPENAI_API_KEY")
CODEX_INVOCATIONS = {
    "provider-key": (
        {"OPENAI_API_KEY": "openai-key"},
        CodexCLIConfig(),
        {"argv": f"{CODEX_ARGV} -", "CODEX_API_KEY": "openai-key"},
    ),
    "gateway-without-a-provider-key": (
        {"HUD_API_KEY": "hud-key"},
        CodexCLIConfig(),
        CODEX_THROUGH_GATEWAY,
    ),
    "gateway-forced-over-a-provider-key": (
        {"HUD_API_KEY": "hud-key", "OPENAI_API_KEY": "openai-key"},
        CodexCLIConfig(gateway=True),
        CODEX_THROUGH_GATEWAY,
    ),
    "ambient-login": ({}, CodexCLIConfig(), {"argv": f"{CODEX_ARGV} -"}),
    "configured-options": (
        {"OPENAI_API_KEY": "openai-key"},
        CodexCLIConfig(model="gpt-5.6", sandbox="read-only", reasoning_effort="max"),
        {
            "argv": "exec --json --ephemeral --skip-git-repo-check --color never --sandbox "
            'read-only --model gpt-5.6 -c model_reasoning_effort="max" -',
            "CODEX_API_KEY": "openai-key",
        },
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
    assert launched(capture, run, CODEX_ROUTING) == expected


CODEX_EXITS = {
    "turn-failed": (
        {
            "stdout": jsonl(
                {"type": "turn.started"},
                {"type": "turn.failed", "error": {"message": "model unavailable"}},
            )
        },
        ("error", None, "[agent loop] RuntimeError: model unavailable"),
    ),
    "nonzero-exit": (
        {"stdout": "", "stderr": "authentication failed", "exit": 1},
        ("error", None, "[agent loop] RuntimeError: authentication failed"),
    ),
    "nonzero-exit-with-a-structured-error": (
        {
            "stdout": jsonl({"type": "error", "message": "gateway rejected streaming"}),
            "stderr": "noisy",
            "exit": 1,
        },
        ("error", None, "[agent loop] RuntimeError: gateway rejected streaming"),
    ),
    "zero-exit-without-turn-completed": (
        {"stdout": jsonl(*CODEX_ITEMS)},
        (
            "error",
            "Implemented and verified.",
            "[agent loop] RuntimeError: codex CLI exited without a turn.completed event",
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


# ─── process-bound inference connections (sandbox lane) ─────────────────

CONNECTION = Connection(
    name="inference",
    capability="ssh",
    url="https://inference.hud.so/v1",
    headers={"Authorization": "Bearer scoped-runtime-token"},
)


def isolated_env(tmp_path: Path, stub: Stub) -> Environment:
    stub.install()
    return workspace_env(
        tmp_path / "ws",
        env=stub.shell_env(),
        allowed_hosts={ANY_HOST},
        require_isolation=True,
        track_files=False,
    )


@pytest.mark.e2e
@pytest.mark.sandbox
async def test_the_claude_cli_reaches_a_bound_connection_without_its_credential(
    hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="hud-key")
    stub = claude_stub(tmp_path)

    run = await run_task(isolated_env(tmp_path, stub), ClaudeCLIAgent(), connections=(CONNECTION,))

    (capture,) = stub.captures()
    env = capture["env"]
    assert run.trace.status == "completed"
    assert (env["ANTHROPIC_BASE_URL"], env["ANTHROPIC_AUTH_TOKEN"]) == (
        CONNECTION.client_url,
        "hud-process-bound",
    )
    assert not {"ANTHROPIC_API_KEY", "ANTHROPIC_CUSTOM_HEADERS"} & env.keys()
    assert "scoped-runtime-token" not in str(capture) and "hud-key" not in str(capture)


@pytest.mark.e2e
@pytest.mark.sandbox
async def test_codex_refuses_a_launcher_script_for_a_bound_connection(
    hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="hud-key")
    stub = codex_stub(tmp_path)

    run = await run_task(isolated_env(tmp_path, stub), CodexCLIAgent(), connections=(CONNECTION,))

    assert (run.trace.error, stub.captures()) == (
        IsStr(regex=r".*/codex is a launcher script, but a process-bound inference connection .*"),
        [],
    )
