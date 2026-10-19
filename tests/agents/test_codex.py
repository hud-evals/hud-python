"""What ``CodexCLIAgent`` hands ``codex exec`` (its prompt, model, settings, where its requests
go and with which credential), what it records from codex's items, and how codex's exit
ends the run. A stub ``codex`` on the workspace's ``PATH`` records what it was handed; the
real CLI runs in the live lane."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from dirty_equals import IsStr

from hud.agents import CodexCLIAgent
from hud.agents.types import CodexCLIConfig, ToolStep
from hud.capabilities import Connection
from hud.environment.egress import ANY_HOST
from tests.agents.cli_stub import (
    CODEX_COMPLETED,
    CODEX_STARTED,
    Stub,
    codex_answer,
    codex_item,
    flag,
    jsonl,
    overrides,
)
from tests.fixtures.environments import run_task, workspace_env

if TYPE_CHECKING:
    from pathlib import Path

    from tests.support import HudEnv, Models


@pytest.fixture(autouse=True)
def signed_in(hud_env: HudEnv) -> None:
    hud_env.set(HUD_API_KEY="hud-key")


async def handed(tmp_path: Path, prompt: str = "Do the task.", **config: Any) -> dict[str, Any]:
    """What one run of ``codex`` was handed: its argv, env and stdin."""
    stub = Stub(tmp_path / "stub", "codex", stdout=codex_answer("done"))
    stub.install()
    env = workspace_env(tmp_path / "ws", env=stub.shell_env())
    run = await run_task(env, CodexCLIAgent(CodexCLIConfig(**config)), prompt=prompt)
    assert run.trace.status == "completed", run.trace.error
    (capture,) = stub.captures()
    return capture


async def test_codex_reads_the_prompt_on_its_stdin(tmp_path: Path) -> None:
    capture = await handed(tmp_path, prompt="Fix the bug.")

    assert capture["stdin"] == "Fix the bug."


@pytest.mark.parametrize(
    ("variables", "config", "route", "credential"),
    [
        pytest.param({"OPENAI_API_KEY": "own-key"}, {}, "provider", "own-key", id="its-own-key"),
        pytest.param({}, {}, "gateway", "hud-key", id="a-hud-key-only"),
        pytest.param(
            {"OPENAI_API_KEY": "own-key"},
            {"gateway": True},
            "gateway",
            "hud-key",
            id="the-gateway-forced-over-its-own-key",
        ),
        pytest.param(
            {"HUD_API_KEY": None}, {}, "its-own-login", None, id="no-key-leaves-its-own-login"
        ),
    ],
)
async def test_codex_is_pointed_where_the_sign_in_routes_it(
    variables: dict[str, str | None],
    config: dict[str, Any],
    route: str,
    credential: str | None,
    models: Models,
    hud_env: HudEnv,
    tmp_path: Path,
) -> None:
    hud_env.set(**variables)

    capture = await handed(tmp_path, **config)

    env, settings = capture["env"], overrides(capture["argv"])
    if route == "gateway":
        assert json.loads(settings["model_providers.hud.base_url"]) == models.url
        assert env[json.loads(settings["model_providers.hud.env_key"])] == credential
        assert '"Trace-Id"=' in settings["model_providers.hud.http_headers"]
    else:
        assert "model_providers.hud.base_url" not in settings
        assert env.get("CODEX_API_KEY") == credential


@pytest.mark.parametrize(
    ("config", "read", "value"),
    [
        pytest.param({"model": "gpt-5.4-mini"}, "--model", "gpt-5.4-mini", id="model"),
        pytest.param({"sandbox": "read-only"}, "--sandbox", "read-only", id="sandbox"),
        pytest.param(
            {"reasoning_effort": "high"}, "model_reasoning_effort", '"high"', id="reasoning-effort"
        ),
        pytest.param({}, "model_reasoning_effort", None, id="no-effort-leaves-the-cli-default"),
        pytest.param(
            {"gateway": True},
            "model_providers.hud.wire_api",
            '"responses"',
            id="the-gateway-speaks-the-responses-api",
        ),
    ],
)
async def test_codex_is_given_the_settings_its_configuration_sets(
    config: dict[str, Any], read: str, value: str | None, tmp_path: Path
) -> None:
    argv = (await handed(tmp_path, **config))["argv"]

    given = flag(argv, read) if read.startswith("--") else overrides(argv).get(read)
    assert given == value


@pytest.mark.parametrize(
    ("item", "step"),
    [
        pytest.param(
            codex_item(
                "command_execution",
                "c",
                command="false",
                aggregated_output="",
                exit_code=1,
                status="completed",
            ),
            ("shell", True, [""]),
            id="a-failed-command",
        ),
        pytest.param(
            codex_item(
                "file_change",
                "f",
                changes=[{"kind": "update", "path": "a.txt"}],
                status="completed",
            ),
            ("apply_patch", False, ["update: a.txt"]),
            id="a-file-change",
        ),
        pytest.param(
            codex_item(
                "mcp_tool_call",
                "m",
                server="tools",
                tool="lookup",
                arguments={"key": "k"},
                result={"content": [{"type": "text", "text": "found"}]},
                status="completed",
            ),
            ("lookup", False, ["found"]),
            id="an-mcp-tool-call",
        ),
        pytest.param(
            codex_item(
                "mcp_tool_call",
                "m",
                server="tools",
                tool="lookup",
                arguments={},
                error={"message": "no such key"},
                status="failed",
            ),
            ("lookup", True, ["no such key"]),
            id="a-failed-mcp-tool-call",
        ),
        pytest.param(
            codex_item("web_search", "w", query="sky", action={"type": "search", "query": "sky"}),
            ("web_search", False, ['{"type":"search","query":"sky"}']),
            id="a-web-search",
        ),
    ],
)
async def test_each_item_codex_ran_lands_on_the_trace_as_a_tool_step(
    item: dict[str, Any], step: tuple[str, bool, list[str]], tmp_path: Path
) -> None:
    stub = Stub(tmp_path / "stub", "codex", stdout=jsonl(CODEX_STARTED, item, CODEX_COMPLETED))
    stub.install()

    run = await run_task(workspace_env(tmp_path / "ws", env=stub.shell_env()), CodexCLIAgent())

    (recorded,) = [s for s in run.trace.steps if isinstance(s, ToolStep)]
    assert recorded.call is not None and recorded.result is not None
    texts = [getattr(block, "text", None) for block in recorded.result.content]
    assert (recorded.call.name, recorded.result.isError, texts) == step


@pytest.mark.parametrize(
    ("behavior", "ending"),
    [
        pytest.param(
            {
                "stdout": jsonl(
                    {"type": "turn.started"},
                    {"type": "turn.failed", "error": {"message": "model unavailable"}},
                )
            },
            ("error", None, "[agent loop] RuntimeError: model unavailable"),
            id="a-failed-turn",
        ),
        pytest.param(
            {"stdout": "", "stderr": "authentication failed", "exit": 1},
            ("error", None, "[agent loop] RuntimeError: authentication failed"),
            id="a-nonzero-exit",
        ),
        pytest.param(
            {
                "stdout": jsonl({"type": "error", "message": "gateway rejected streaming"}),
                "stderr": "noisy",
                "exit": 1,
            },
            ("error", None, "[agent loop] RuntimeError: gateway rejected streaming"),
            id="a-nonzero-exit-with-a-structured-error",
        ),
        pytest.param(
            {
                "stdout": jsonl(
                    CODEX_STARTED,
                    codex_item("agent_message", "message-1", text="Implemented and verified."),
                )
            },
            (
                "error",
                "Implemented and verified.",
                "[agent loop] RuntimeError: codex CLI exited without a turn.completed event",
            ),
            id="a-zero-exit-without-turn-completed",
        ),
    ],
)
async def test_how_codex_exits_decides_how_the_run_ends(
    behavior: dict[str, Any], ending: tuple[str, str | None, str], tmp_path: Path
) -> None:
    stub = Stub(tmp_path / "stub", "codex", **behavior)
    stub.install()

    run = await run_task(workspace_env(tmp_path / "ws", env=stub.shell_env()), CodexCLIAgent())

    assert (run.trace.status, run.trace.content, run.trace.error) == ending


@pytest.mark.e2e
@pytest.mark.sandbox
async def test_codex_refuses_a_launcher_script_for_a_process_bound_connection(
    tmp_path: Path,
) -> None:
    connection = Connection(
        name="inference",
        capability="ssh",
        url="https://inference.hud.so/v1",
        headers={"Authorization": "Bearer scoped-runtime-token"},
    )
    stub = Stub(tmp_path / "stub", "codex", stdout=codex_answer("Implemented and verified."))
    stub.install()
    env = workspace_env(
        tmp_path / "ws",
        env=stub.shell_env(),
        allowed_hosts={ANY_HOST},
        require_isolation=True,
        track_files=False,
    )

    run = await run_task(env, CodexCLIAgent(), connections=(connection,))

    assert (run.trace.error, stub.captures()) == (
        IsStr(regex=r".*/codex is a launcher script, but a process-bound inference connection .*"),
        [],
    )
