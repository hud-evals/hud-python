"""What only ``ClaudeCLIAgent`` does: what it hands the ``claude`` CLI (prompt, tools, MCP
servers, where its requests go and which model each tier uses), how the CLI's exit ends
the run, and the computer bridge it serves the CLI. A stand-in ``claude`` on the
workspace's ``PATH`` records what it was handed; the real CLI runs in the live lane."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from fastmcp import FastMCP

from hud.agents import ClaudeCLIAgent
from hud.agents.types import ClaudeCLIConfig
from hud.capabilities import Capability, Connection
from hud.environment.egress import ANY_HOST
from tests.fixtures import clis
from tests.fixtures.asgi import mcp_server
from tests.fixtures.environments import run_task, workspace_env
from tests.fixtures.screen import fake_screen

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import HudEnv
    from tests.fixtures.models import Models


@pytest.fixture(autouse=True)
def signed_in(hud_env: HudEnv) -> None:
    hud_env.set(HUD_API_KEY="hud-key")


async def test_the_cli_is_given_the_prompt_its_allowed_tools_and_the_environments_servers(
    tmp_path: Path,
) -> None:
    lookups = FastMCP("tools")
    lookups.tool(name="lookup", description="Look a key up.")(lambda key: key)
    stub = clis.Stub(tmp_path / "stub", "claude", stdout=clis.answer("claude", "done"))
    stub.install()

    async with mcp_server(lookups) as server:
        env = workspace_env(tmp_path / "ws", env=stub.shell_env(), capabilities=(server,))
        await run_task(env, ClaudeCLIAgent(), prompt="Fix the bug.")

    ((argv, stdin, files),) = [(c["argv"], c["stdin"], c["files"]) for c in stub.captures()]
    allowed = [argv[index + 1] for index, arg in enumerate(argv) if arg == "--allowedTools"]
    assert clis.prompt("claude", stdin) == "Fix the bug."
    assert allowed == ClaudeCLIConfig().allowed_tools
    assert list(json.loads(files[".hud_mcp_config.json"])["mcpServers"]) == ["tools"]


@pytest.mark.parametrize(
    ("own_key", "gateway", "via_gateway", "credential"),
    [
        ("own-key", False, False, "own-key"),
        (None, False, True, "hud-key"),
        ("own-key", True, True, "hud-key"),
    ],
    ids=["its-own-key", "only-a-hud-key", "the-gateway-forced-over-its-own-key"],
)
async def test_the_cli_is_pointed_at_the_provider_with_its_own_key_or_else_the_gateway(
    own_key: str | None,
    gateway: bool,
    via_gateway: bool,
    credential: str,
    models: Models,
    hud_env: HudEnv,
    tmp_path: Path,
) -> None:
    hud_env.set(ANTHROPIC_API_KEY=own_key)
    stub = clis.Stub(tmp_path / "stub", "claude", stdout=clis.answer("claude", "done"))
    stub.install()

    await run_task(
        workspace_env(tmp_path / "ws", env=stub.shell_env()),
        ClaudeCLIAgent(ClaudeCLIConfig(gateway=gateway)),
    )

    (env,) = [capture["env"] for capture in stub.captures()]
    assert env.get("ANTHROPIC_BASE_URL") == (models.url if via_gateway else None)
    assert env.get("ANTHROPIC_API_KEY") == credential
    assert ("Trace-Id: " in env.get("ANTHROPIC_CUSTOM_HEADERS", "")) is via_gateway


async def test_every_model_tier_and_subagents_use_the_configured_model(tmp_path: Path) -> None:
    stub = clis.Stub(tmp_path / "stub", "claude", stdout=clis.answer("claude", "done"))
    stub.install()

    await run_task(
        workspace_env(tmp_path / "ws", env=stub.shell_env()),
        ClaudeCLIAgent(ClaudeCLIConfig(model="claude-opus-5", gateway=True)),
    )

    (env,) = [capture["env"] for capture in stub.captures()]
    tiers = (
        "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "CLAUDE_CODE_SUBAGENT_MODEL",
    )
    assert {env[name] for name in tiers} == {"claude-opus-5"}


WRITE = clis.claude_assistant(
    {"type": "text", "text": "editing"},
    {"type": "tool_use", "id": "tool-1", "name": "Write", "input": {"file_path": "a.txt"}},
    stop="tool_use",
)
DONE = clis.claude_assistant({"type": "text", "text": "done"})


@pytest.mark.parametrize(
    ("stdout", "stderr", "exit", "status", "content", "error"),
    [
        ("", "boom", 1, "error", None, "boom"),
        (clis.answer("claude", "done"), "transport failed", 1, "error", "done", "transport failed"),
        (clis.answer("claude", "done"), "", 1, "completed", "done", None),
        (clis.jsonl(DONE), "", 0, "error", "done", "claude CLI exited without a result event"),
        (
            clis.jsonl(DONE, clis.claude_result("rate limited upstream", is_error=True)),
            "",
            0,
            "error",
            "rate limited upstream",
            "rate limited upstream",
        ),
        (
            clis.jsonl(WRITE, clis.claude_result()),
            "",
            0,
            "error",
            "done",
            "claude CLI exited without results for tool calls: tool-1",
        ),
    ],
    ids=[
        "a-nonzero-exit-without-output",
        "a-nonzero-exit-after-a-result-with-stderr",
        "a-nonzero-exit-after-a-result-without-stderr",
        "a-zero-exit-without-a-result",
        "a-result-reporting-an-error",
        "a-tool-call-left-without-a-result",
    ],
)
async def test_how_the_cli_exits_decides_how_the_run_ends(
    stdout: str,
    stderr: str,
    exit: int,
    status: str,
    content: str | None,
    error: str | None,
    tmp_path: Path,
) -> None:
    stub = clis.Stub(tmp_path / "stub", "claude", stdout=stdout, stderr=stderr, exit=exit)
    stub.install()

    run = await run_task(workspace_env(tmp_path / "ws", env=stub.shell_env()), ClaudeCLIAgent())

    assert (run.trace.status, run.trace.content) == (status, content)
    assert run.trace.error == (None if error is None else f"[agent loop] RuntimeError: {error}")


async def test_mcp_servers_reach_the_cli_in_a_config_removed_after_the_run(
    tmp_path: Path,
) -> None:
    stub = clis.Stub(tmp_path / "stub", "claude", stdout=clis.answer("claude", "done"))
    stub.install()
    servers = (
        Capability.mcp(name="database", url="http://db.internal:8000/mcp", auth_token="secret"),
        Capability.mcp(name="search", url="http://search.internal:8000/sse", transport="sse"),
    )

    await run_task(
        workspace_env(tmp_path / "ws", env=stub.shell_env(), capabilities=servers),
        ClaudeCLIAgent(),
    )

    (capture,) = stub.captures()
    assert capture["argv"][-2:] == ["--mcp-config", ".hud_mcp_config.json"]
    configured = json.loads(capture["files"][".hud_mcp_config.json"])["mcpServers"]
    assert {name: (server["type"], server["url"]) for name, server in configured.items()} == {
        "database": ("http", "http://db.internal:8000/mcp"),
        "search": ("sse", "http://search.internal:8000/sse"),
    }
    assert configured["database"]["headers"]["Authorization"] == "Bearer secret"
    assert list((tmp_path / "ws").iterdir()) == []


async def test_the_cli_drives_the_screen_through_the_computer_bridge(tmp_path: Path) -> None:
    actions: list[dict[str, Any]] = [
        {"action": "left_click", "coordinate": [3, 4]},
        {"action": "frobnicate"},
    ]
    stub = clis.Stub(
        tmp_path / "stub", "claude", stdout=clis.answer("claude", "done"), computer=actions
    )
    stub.install()

    async with fake_screen() as screen:
        env = workspace_env(
            tmp_path / "ws", env=stub.shell_env(), capabilities=(Capability.rfb(url=screen.url),)
        )
        run = await run_task(env, ClaudeCLIAgent())

    (capture,) = stub.captures()
    assert run.trace.status == "completed"
    assert capture["computer"] == [
        {"isError": False, "content": ["image"]},
        {"isError": True, "content": ["unsupported computer action: 'frobnicate'"]},
    ]
    assert [(e.x, e.y, e.buttons) for e in screen.pointer_events()] == [
        (3, 4, 0),
        (3, 4, 1),
        (3, 4, 0),
    ]


async def test_an_mcp_server_named_like_the_computer_bridge_is_refused(tmp_path: Path) -> None:
    stub = clis.Stub(tmp_path / "stub", "claude", stdout=clis.answer("claude", "done"))
    stub.install()

    async with fake_screen() as screen:
        clash = (
            Capability.rfb(url=screen.url),
            Capability.mcp(name="computer-use", url="http://tools.internal:8000/mcp"),
        )
        env = workspace_env(tmp_path / "ws", env=stub.shell_env(), capabilities=clash)
        run = await run_task(env, ClaudeCLIAgent())

    assert (run.trace.error, stub.captures()) == (
        "[agent loop] RuntimeError: duplicate MCP server name 'computer-use'",
        [],
    )


@pytest.mark.e2e
@pytest.mark.sandbox
async def test_the_cli_reaches_a_process_bound_connection_without_its_credential(
    tmp_path: Path,
) -> None:
    connection = Connection(
        name="inference",
        capability="ssh",
        url="https://inference.hud.so/v1",
        headers={"Authorization": "Bearer scoped-runtime-token"},
    )
    stub = clis.Stub(tmp_path / "stub", "claude", stdout=clis.answer("claude", "done"))
    stub.install()
    env = workspace_env(
        tmp_path / "ws",
        env=stub.shell_env(),
        allowed_hosts={ANY_HOST},
        require_isolation=True,
        track_files=False,
    )

    run = await run_task(env, ClaudeCLIAgent(), connections=(connection,))

    (capture,) = stub.captures()
    assert run.trace.status == "completed"
    assert (capture["env"]["ANTHROPIC_BASE_URL"], capture["env"]["ANTHROPIC_AUTH_TOKEN"]) == (
        connection.client_url,
        "hud-process-bound",
    )
    assert not {"ANTHROPIC_API_KEY", "ANTHROPIC_CUSTOM_HEADERS"} & capture["env"].keys()
    assert "scoped-runtime-token" not in str(capture) and "hud-key" not in str(capture)
