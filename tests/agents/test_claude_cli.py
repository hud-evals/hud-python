"""What ``ClaudeCLIAgent`` hands the ``claude`` CLI (its prompt, tools, model, flags, where its
requests go and with which credential), and how the CLI's exit ends the run. A stub
``claude`` on the workspace's ``PATH`` records what it was handed; the real CLI runs in
the live lane."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from fastmcp import FastMCP

from hud.agents import ClaudeCLIAgent
from hud.agents.types import ClaudeCLIConfig
from hud.capabilities import Capability, Connection
from hud.environment.egress import ANY_HOST
from tests.agents.cli_stub import (
    Stub,
    claude_answer,
    claude_assistant,
    claude_result,
    flag,
    jsonl,
)
from tests.fixtures.environments import run_task, workspace_env
from tests.support import fake_screen, mcp_server

if TYPE_CHECKING:
    from pathlib import Path

    from tests.support import HudEnv, Models


@pytest.fixture(autouse=True)
def signed_in(hud_env: HudEnv) -> None:
    hud_env.set(HUD_API_KEY="hud-key")


async def handed(
    tmp_path: Path,
    capabilities: tuple[Capability, ...] = (),
    prompt: str = "Do the task.",
    **config: Any,
) -> dict[str, Any]:
    """What one run of ``claude`` was handed: its argv, env, stdin and ``.hud_*`` files."""
    stub = Stub(tmp_path / "stub", "claude", stdout=claude_answer("done"))
    stub.install()
    env = workspace_env(tmp_path / "ws", env=stub.shell_env(), capabilities=capabilities)
    run = await run_task(env, ClaudeCLIAgent(ClaudeCLIConfig(**config)), prompt=prompt)
    assert run.trace.status == "completed", run.trace.error
    (capture,) = stub.captures()
    return capture


def lookup_server() -> FastMCP:
    server = FastMCP("tools")
    server.tool(name="lookup", description="Look a key up.")(lambda key: key)
    return server


async def test_the_cli_is_given_the_prompt_its_allowed_tools_and_the_environments_servers(
    tmp_path: Path,
) -> None:
    async with mcp_server(lookup_server()) as server:
        capture = await handed(tmp_path, (server,), prompt="Fix the bug.")

    argv = capture["argv"]
    (message,) = [json.loads(line) for line in capture["stdin"].splitlines()]
    allowed = [argv[index + 1] for index, arg in enumerate(argv) if arg == "--allowedTools"]
    servers = json.loads(capture["files"][".hud_mcp_config.json"])["mcpServers"]
    assert message["message"]["content"][0]["text"] == "Fix the bug."
    assert (allowed, list(servers)) == (ClaudeCLIConfig().allowed_tools, ["tools"])


@pytest.mark.parametrize(
    ("variables", "config", "base_url", "key", "traced"),
    [
        pytest.param(
            {"ANTHROPIC_API_KEY": "own-key"}, {}, None, "own-key", False, id="its-own-key"
        ),
        pytest.param({}, {}, "<gateway>", "hud-key", True, id="a-hud-key-only"),
        pytest.param(
            {"ANTHROPIC_API_KEY": "own-key"},
            {"gateway": True},
            "<gateway>",
            "hud-key",
            True,
            id="the-gateway-forced-over-its-own-key",
        ),
    ],
)
async def test_the_cli_is_pointed_where_the_sign_in_routes_it(
    variables: dict[str, str],
    config: dict[str, Any],
    base_url: str | None,
    key: str,
    traced: bool,
    models: Models,
    hud_env: HudEnv,
    tmp_path: Path,
) -> None:
    hud_env.set(**variables)

    env = (await handed(tmp_path, **config))["env"]

    headers = env.get("ANTHROPIC_CUSTOM_HEADERS", "")
    assert env.get("ANTHROPIC_BASE_URL") == (models.url if base_url else None)
    assert (env.get("ANTHROPIC_API_KEY"), "Trace-Id: " in headers) == (key, traced)


async def test_the_model_is_named_for_every_tier(tmp_path: Path) -> None:
    env = (await handed(tmp_path, model="claude-opus-5", gateway=True))["env"]

    tiers = (
        "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "CLAUDE_CODE_SUBAGENT_MODEL",
    )
    assert {env[name] for name in tiers} == {"claude-opus-5"}


@pytest.mark.parametrize(
    ("config", "name", "value"),
    [
        pytest.param({"reasoning_effort": "max"}, "--effort", "max", id="reasoning-effort"),
        pytest.param({}, "--effort", None, id="no-effort-leaves-the-cli-default"),
        pytest.param(
            {"system_prompt": "Be brief."}, "--system-prompt", "Be brief.", id="system-prompt"
        ),
        pytest.param(
            {"permission_mode": "acceptEdits"},
            "--permission-mode",
            "acceptEdits",
            id="permission-mode",
        ),
        pytest.param({"max_steps": 2}, "--max-turns", "2", id="the-step-budget"),
        pytest.param({}, "--max-turns", None, id="no-step-budget"),
    ],
)
async def test_the_cli_is_given_the_flags_its_configuration_sets(
    config: dict[str, Any], name: str, value: str | None, tmp_path: Path
) -> None:
    capture = await handed(tmp_path, **config)

    assert flag(capture["argv"], name) == value


WRITE = claude_assistant(
    {"type": "text", "text": "editing"},
    {"type": "tool_use", "id": "tool-1", "name": "Write", "input": {"file_path": "a.txt"}},
    stop="tool_use",
)
DONE = claude_assistant({"type": "text", "text": "done"})


@pytest.mark.parametrize(
    ("behavior", "ending"),
    [
        pytest.param(
            {"stdout": "", "stderr": "boom", "exit": 1},
            ("error", None, "[agent loop] RuntimeError: boom"),
            id="a-nonzero-exit-without-output",
        ),
        pytest.param(
            {"stdout": claude_answer("done"), "stderr": "transport failed", "exit": 1},
            ("error", "done", "[agent loop] RuntimeError: transport failed"),
            id="a-nonzero-exit-after-a-result-with-stderr",
        ),
        pytest.param(
            {"stdout": claude_answer("done"), "exit": 1},
            ("completed", "done", None),
            id="a-nonzero-exit-after-a-result-without-stderr",
        ),
        pytest.param(
            {"stdout": jsonl(DONE)},
            (
                "error",
                "done",
                "[agent loop] RuntimeError: claude CLI exited without a result event",
            ),
            id="a-zero-exit-without-a-result",
        ),
        pytest.param(
            {"stdout": jsonl(DONE, claude_result("rate limited upstream", is_error=True))},
            ("error", "rate limited upstream", "[agent loop] RuntimeError: rate limited upstream"),
            id="a-result-reporting-an-error",
        ),
        pytest.param(
            {"stdout": jsonl(WRITE, claude_result())},
            (
                "error",
                "done",
                "[agent loop] RuntimeError: "
                "claude CLI exited without results for tool calls: tool-1",
            ),
            id="a-tool-call-left-without-a-result",
        ),
    ],
)
async def test_how_the_cli_exits_decides_how_the_run_ends(
    behavior: dict[str, Any], ending: tuple[str, str | None, str | None], tmp_path: Path
) -> None:
    stub = Stub(tmp_path / "stub", "claude", **behavior)
    stub.install()

    run = await run_task(workspace_env(tmp_path / "ws", env=stub.shell_env()), ClaudeCLIAgent())

    assert (run.trace.status, run.trace.content, run.trace.error) == ending


async def test_mcp_servers_reach_the_cli_in_a_config_removed_after_the_run(
    tmp_path: Path,
) -> None:
    servers = (
        Capability.mcp(name="database", url="http://db.internal:8000/mcp", auth_token="secret"),
        Capability.mcp(name="search", url="http://search.internal:8000/sse", transport="sse"),
    )

    capture = await handed(tmp_path, servers)

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
    assert list((tmp_path / "ws").iterdir()) == []


async def test_the_cli_drives_the_screen_through_the_computer_bridge(tmp_path: Path) -> None:
    stub = Stub(
        tmp_path / "stub",
        "claude",
        stdout=claude_answer("done"),
        computer=[{"action": "left_click", "coordinate": [3, 4]}, {"action": "frobnicate"}],
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
    stub = Stub(tmp_path / "stub", "claude", stdout=claude_answer("done"))
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
    stub = Stub(tmp_path / "stub", "claude", stdout=claude_answer("done"))
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
