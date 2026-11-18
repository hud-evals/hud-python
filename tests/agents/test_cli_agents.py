"""What the coding-CLI agents do with their CLI: one CLI per concurrent rollout, the agent
timeout stopping it, how its exit ends the run, and what Claude Code, Codex and Pi each
reach and record. A stand-in on the workspace's ``PATH`` plays each CLI."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from fastmcp import FastMCP
from pydantic import ValidationError

from hud import Environment
from hud.agents import Agent, ClaudeCLIAgent, CodexCLIAgent, PiAgent
from hud.agents.types import AgentStep, ClaudeCLIConfig, PiConfig, ToolStep
from hud.capabilities import Capability
from hud.environment.egress import ANY_HOST
from hud.environment.workspace import usable_bwrap
from hud.eval.run import rollout
from hud.runtime import LocalRuntime, Shared
from hud.types import AgentType
from tests.conftest import SANDBOX
from tests.fixtures import clis
from tests.fixtures.asgi import serve_asgi
from tests.fixtures.environments import workspace_task
from tests.fixtures.screen import fake_screen

if TYPE_CHECKING:
    from hud import Run
    from tests.conftest import HudEnv

UNSANDBOXED = pytest.mark.skipif(
    usable_bwrap() is not None, reason="this host sandboxes the workspace"
)
# (agent type, the executable it launches).
EVERY_CLI = pytest.mark.parametrize(
    ("agent_type", "cli"),
    [
        pytest.param("claude_cli", "claude", id="claude_cli"),
        pytest.param("codex_cli", "codex", id="codex_cli"),
        pytest.param("pi", "pi", id="pi"),
    ],
)
# The variable each CLI reads its model key from.
MODEL_KEY = {"claude": "ANTHROPIC_API_KEY", "codex": "INFERENCE_API_KEY", "pi": "INFERENCE_API_KEY"}
# The variable naming the directory each CLI reads its login and settings from.
CONFIG_HOME = {"claude": "CLAUDE_CONFIG_DIR", "codex": "CODEX_HOME", "pi": "PI_CODING_AGENT_DIR"}
# Where a workspace runs: unsandboxed where the host gives no sandbox, and in the
# sandbox lane, where its CLI reaches its model through a process-bound relay.
BACKINGS = pytest.mark.parametrize(
    "backing",
    [
        pytest.param({}, id="host", marks=UNSANDBOXED),
        pytest.param(
            {"allowed_hosts": {ANY_HOST}, "require_isolation": True, "track_files": False},
            id="sandbox",
            marks=SANDBOX,
        ),
    ],
)


@pytest.fixture(autouse=True)
def signed_in(hud_env: HudEnv) -> None:
    hud_env.set(HUD_API_KEY="hud-key")


@EVERY_CLI
async def test_one_agent_drives_concurrent_rollouts_each_with_its_own_cli(
    agent_type: str, cli: str, tmp_path: Path
) -> None:
    stub = clis.Stub(tmp_path / "ws" / ".stub", cli, stdout=clis.answer(cli, "done"))
    stub.install()
    task, agent = workspace_task(tmp_path / "ws", env=stub.shell_env()), AgentType(agent_type).cls()

    jobs = await asyncio.gather(task(prompt="first").run(agent), task(prompt="second").run(agent))

    assert [trace.status for job in jobs for trace in job.traces] == ["completed", "completed"]
    assert sorted(clis.prompt(cli, capture["stdin"]) for capture in stub.captures()) == [
        "first",
        "second",
    ]


@EVERY_CLI
async def test_the_agent_timeout_ends_the_run_and_the_cli_while_the_workspace_lives_on(
    agent_type: str, cli: str, tmp_path: Path
) -> None:
    """A pooled workspace outlives the run, so only the agent can end the CLI it started."""
    kind = AgentType(agent_type).cls
    stub = clis.Stub(tmp_path / "ws" / ".stub", cli, stdout=clis.answer(cli, "done"))
    stub.install()
    started = time.monotonic()
    task = workspace_task(tmp_path / "ws", env=stub.shell_env())
    await task().run(kind())
    # One second plus twice a measured full run, so the hung CLI has started before the
    # timeout lands.
    timeout = 1.0 + 2 * (time.monotonic() - started)
    hung = clis.Stub(
        tmp_path / "ws" / ".stub",
        cli,
        stdout=clis.answer(cli, "done").splitlines()[0] + "\n",
        hang=True,
    )
    hung.install()
    task = workspace_task(tmp_path / "ws", env=hung.shell_env())

    async with Shared(LocalRuntime(task.env), width=1) as pool:
        trace = await rollout(
            task(),
            kind(kind.config_cls(timeout_seconds=timeout)),
            runtime=pool,
        )
        (capture,) = hung.captures()
        exited = clis.process_exited(capture["pid"])

    assert (trace.status, trace.stop_reason, exited) == ("error", "timeout", True)


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
    stub = clis.Stub(tmp_path / "ws" / ".stub", "claude", stdout=stdout, stderr=stderr, exit=exit)
    stub.install()

    task = workspace_task(tmp_path / "ws", env=stub.shell_env())
    (trace,) = (await task().run(ClaudeCLIAgent())).traces

    assert (trace.status, trace.content) == (status, content)
    assert trace.error == (None if error is None else f"[agent loop] RuntimeError: {error}")


async def test_a_cli_run_that_reaches_its_turn_limit_stops_at_max_steps(tmp_path: Path) -> None:
    stub = clis.Stub(
        tmp_path / "ws" / ".stub",
        "claude",
        stdout=clis.jsonl(DONE, clis.CLAUDE_TURN_LIMIT),
        exit=1,
    )
    stub.install()

    task = workspace_task(tmp_path / "ws", env=stub.shell_env())
    (trace,) = (await task().run(ClaudeCLIAgent(ClaudeCLIConfig(max_steps=1)))).traces

    assert (trace.status, trace.stop_reason, trace.content, trace.error) == (
        "completed",
        "max_steps",
        "done",
        None,
    )


@BACKINGS
async def test_the_cli_drives_the_screen_through_the_computer_bridge(
    backing: dict[str, Any], tmp_path: Path
) -> None:
    actions: list[dict[str, Any]] = [
        {"action": "left_click", "coordinate": [3, 4]},
        {"action": "frobnicate"},
    ]
    stub = clis.Stub(
        tmp_path / "ws" / ".stub", "claude", stdout=clis.answer("claude", "done"), computer=actions
    )
    stub.install()

    async with fake_screen() as screen:
        task = workspace_task(
            tmp_path / "ws",
            env=stub.shell_env(),
            capabilities=(Capability.rfb(url=screen.url),),
            **backing,
        )
        (trace,) = (await task().run(ClaudeCLIAgent())).traces

    (capture,) = stub.captures()
    assert trace.status == "completed"
    assert capture["computer"] == [
        {"isError": False, "content": ["image"]},
        {"isError": True, "content": ["unsupported computer action: 'frobnicate'"]},
    ]
    assert [(e.x, e.y, e.buttons) for e in screen.pointer_events()] == [
        (3, 4, 0),
        (3, 4, 1),
        (3, 4, 0),
    ]


@UNSANDBOXED
@EVERY_CLI
async def test_without_a_sandbox_the_cli_holds_its_model_key(
    agent_type: str, cli: str, tmp_path: Path
) -> None:
    stub = clis.Stub(tmp_path / "ws" / ".stub", cli, stdout=clis.answer(cli, "done"))
    stub.install()
    task = workspace_task(tmp_path / "ws", env=stub.shell_env())

    trace = await rollout(
        task(),
        AgentType(agent_type).cls(),
        runtime=LocalRuntime(task.env),
    )

    (capture,) = stub.captures()
    assert trace.status == "completed"
    assert "hud-key" in capture["env"].values()


@UNSANDBOXED
@EVERY_CLI
async def test_a_cli_given_the_hud_key_reads_no_login_or_settings_from_the_users_home(
    agent_type: str, cli: str, tmp_path: Path
) -> None:
    stub = clis.Stub(tmp_path / "ws" / ".stub", cli, stdout=clis.answer(cli, "done"))
    stub.install()
    shell_env = stub.shell_env()
    task = workspace_task(tmp_path / "ws", env=shell_env)

    trace = await rollout(task(), AgentType(agent_type).cls(), runtime=LocalRuntime(task.env))

    (capture,) = stub.captures()
    config_home = Path(capture["env"][CONFIG_HOME[cli]])
    assert trace.status == "completed"
    assert not config_home.is_relative_to(shell_env["HOME"])
    assert not await asyncio.to_thread(config_home.exists)


@pytest.mark.e2e
@pytest.mark.sandbox
@EVERY_CLI
async def test_in_a_sandbox_the_cli_is_handed_the_relay_and_never_the_hud_key(
    agent_type: str, cli: str, tmp_path: Path
) -> None:
    stub = clis.Stub(tmp_path / "ws" / ".stub", cli, stdout=clis.answer(cli, "done"))
    stub.install()
    env = Environment("ws")
    env.workspace(
        tmp_path / "ws",
        guest_path=str(tmp_path / "ws"),
        env=stub.shell_env(),
        allowed_hosts={ANY_HOST},
        require_isolation=True,
        track_files=False,
    )

    @env.template()
    async def task():
        yield "Do the task."
        yield 1.0

    trace = await rollout(task(), AgentType(agent_type).cls(), runtime=LocalRuntime(env))

    (capture,) = stub.captures()
    assert trace.status == "completed"
    assert capture["env"][MODEL_KEY[cli]] == "hud-process-bound"
    assert "hud-key" not in str(capture)


@pytest.mark.e2e
@pytest.mark.sandbox
@EVERY_CLI
async def test_cli_rollouts_one_after_another_on_a_shared_sandbox_are_each_handed_the_relay(
    agent_type: str, cli: str, tmp_path: Path
) -> None:
    stub = clis.Stub(tmp_path / "ws" / ".stub", cli, stdout=clis.answer(cli, "done"))
    stub.install()
    task = workspace_task(
        tmp_path / "ws",
        env=stub.shell_env(),
        allowed_hosts={ANY_HOST},
        require_isolation=True,
        track_files=False,
    )

    async with Shared(LocalRuntime(task.env), width=1) as runtime:
        agent = AgentType(agent_type).cls()
        traces = [await rollout(task(), agent, runtime=runtime) for _ in range(2)]

    assert [trace.status for trace in traces] == ["completed", "completed"]
    assert [capture["env"][MODEL_KEY[cli]] for capture in stub.captures()] == [
        "hud-process-bound"
    ] * 2


@pytest.mark.e2e
@pytest.mark.sandbox
async def test_in_a_sandbox_an_agent_on_its_provider_key_is_refused_rather_than_rerouted(
    hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(ANTHROPIC_API_KEY="anthropic-key")
    stub = clis.Stub(tmp_path / "ws" / ".stub", "claude", stdout=clis.answer("claude", "done"))
    stub.install()
    task = workspace_task(
        tmp_path / "ws",
        env=stub.shell_env(),
        allowed_hosts={ANY_HOST},
        require_isolation=True,
        track_files=False,
    )

    trace = await rollout(task(), ClaudeCLIAgent(), runtime=LocalRuntime(task.env))

    assert trace.error is not None
    assert "isn't given the agent's model key" in trace.error
    assert stub.captures() == []


class SeesConnections(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[str] | None = None

    async def __call__(self, run: Run) -> None:
        self.seen = list(run.connections)


@pytest.mark.e2e
@pytest.mark.sandbox
async def test_in_a_sandbox_an_agent_that_runs_no_cli_is_given_no_relay(tmp_path: Path) -> None:
    task = workspace_task(
        tmp_path / "ws", allowed_hosts={ANY_HOST}, require_isolation=True, track_files=False
    )
    agent = SeesConnections()

    await rollout(task(), agent, runtime=LocalRuntime(task.env))

    assert agent.seen == []


async def test_a_sandboxed_workspace_that_cannot_relay_is_refused_the_model_key() -> None:
    env = Environment(
        "remote", capabilities=(Capability.ssh(url="ssh://127.0.0.1:1", host_pubkey="key"),)
    )

    @env.template()
    async def task():
        yield "Do the task."
        yield 1.0

    trace = await rollout(task(), ClaudeCLIAgent(), runtime=LocalRuntime(env))

    assert trace.error is not None
    assert "isn't given the agent's model key" in trace.error


async def test_claude_code_runs_with_the_variables_its_config_adds(tmp_path: Path) -> None:
    stub = clis.Stub(tmp_path / "ws" / ".stub", "claude", stdout=clis.answer("claude", "done"))
    stub.install()
    task = workspace_task(tmp_path / "ws", env=stub.shell_env())
    agent = ClaudeCLIAgent(ClaudeCLIConfig(env={"ANTHROPIC_DEFAULT_HAIKU_MODEL": "haiku"}))

    await rollout(task(), agent, runtime=LocalRuntime(task.env))

    (capture,) = stub.captures()
    assert capture["env"]["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "haiku"


def test_claude_code_config_refuses_a_variable_name_no_shell_can_assign() -> None:
    with pytest.raises(ValidationError, match="HTTP-PROXY"):
        ClaudeCLIConfig(env={"HTTP-PROXY": "http://proxy"})


@pytest.mark.parametrize("variable", ["ANTHROPIC_API_KEY", "anthropic_api_key"])
async def test_claude_code_refuses_config_variables_the_agent_sets(
    variable: str, tmp_path: Path
) -> None:
    stub = clis.Stub(tmp_path / "ws" / ".stub", "claude", stdout=clis.answer("claude", "done"))
    stub.install()
    task = workspace_task(tmp_path / "ws", env=stub.shell_env())
    agent = ClaudeCLIAgent(ClaudeCLIConfig(env={variable: "key"}))

    (trace,) = (await task().run(agent)).traces

    assert trace.error is not None
    assert variable in trace.error
    assert stub.captures() == []


@pytest.mark.parametrize(
    ("item", "name", "is_error", "texts"),
    [
        (
            clis.codex_item(
                "command_execution",
                "c",
                command="false",
                aggregated_output="",
                exit_code=1,
                status="completed",
            ),
            "shell",
            True,
            [""],
        ),
        (
            clis.codex_item(
                "file_change",
                "f",
                changes=[{"kind": "update", "path": "a.txt"}],
                status="completed",
            ),
            "apply_patch",
            False,
            ["update: a.txt"],
        ),
        (
            clis.codex_item(
                "mcp_tool_call",
                "m",
                server="tools",
                tool="lookup",
                arguments={"key": "k"},
                result={"content": [{"type": "text", "text": "found"}]},
                status="completed",
            ),
            "lookup",
            False,
            ["found"],
        ),
        (
            clis.codex_item(
                "mcp_tool_call",
                "m",
                server="tools",
                tool="lookup",
                arguments={},
                error={"message": "no such key"},
                status="failed",
            ),
            "lookup",
            True,
            ["no such key"],
        ),
        (
            clis.codex_item(
                "web_search", "w", query="sky", action={"type": "search", "query": "sky"}
            ),
            "web_search",
            False,
            ['{"type":"search","query":"sky"}'],
        ),
    ],
    ids=[
        "a-failed-command",
        "a-file-change",
        "an-mcp-tool-call",
        "a-failed-mcp-tool-call",
        "a-web-search",
    ],
)
async def test_each_item_codex_ran_lands_on_the_trace_as_a_tool_step(
    item: dict[str, Any], name: str, is_error: bool, texts: list[str], tmp_path: Path
) -> None:
    stdout = clis.jsonl(clis.CODEX_STARTED, item, clis.CODEX_COMPLETED)
    stub = clis.Stub(tmp_path / "ws" / ".stub", "codex", stdout=stdout)
    stub.install()

    task = workspace_task(tmp_path / "ws", env=stub.shell_env())
    (trace,) = (await task().run(CodexCLIAgent())).traces

    (recorded,) = [step for step in trace.steps if isinstance(step, ToolStep)]
    assert recorded.call is not None and recorded.result is not None
    sent = [getattr(block, "text", None) for block in recorded.result.content]
    assert (recorded.call.name, recorded.result.isError, sent) == (name, is_error, texts)


@pytest.mark.parametrize(
    ("stdout", "stderr", "exit", "content", "error"),
    [
        (
            clis.jsonl(
                {"type": "turn.started"},
                {"type": "turn.failed", "error": {"message": "model unavailable"}},
            ),
            "",
            0,
            None,
            "model unavailable",
        ),
        ("", "authentication failed", 1, None, "authentication failed"),
        (
            clis.jsonl({"type": "error", "message": "gateway rejected streaming"}),
            "noisy",
            1,
            None,
            "gateway rejected streaming",
        ),
        (
            clis.jsonl(
                clis.CODEX_STARTED,
                clis.codex_item("agent_message", "message-1", text="Implemented and verified."),
            ),
            "",
            0,
            "Implemented and verified.",
            "codex CLI exited without a turn.completed event",
        ),
    ],
    ids=[
        "a-failed-turn",
        "a-nonzero-exit",
        "a-nonzero-exit-with-a-structured-error",
        "a-zero-exit-without-turn-completed",
    ],
)
async def test_how_codex_exits_decides_how_the_run_ends(
    stdout: str, stderr: str, exit: int, content: str | None, error: str, tmp_path: Path
) -> None:
    stub = clis.Stub(tmp_path / "ws" / ".stub", "codex", stdout=stdout, stderr=stderr, exit=exit)
    stub.install()

    task = workspace_task(tmp_path / "ws", env=stub.shell_env())
    (trace,) = (await task().run(CodexCLIAgent())).traces

    assert (trace.status, trace.content, trace.error) == (
        "error",
        content,
        f"[agent loop] RuntimeError: {error}",
    )


@pytest.mark.e2e
@pytest.mark.sandbox
async def test_codex_refuses_a_launcher_script_for_a_process_bound_connection(
    tmp_path: Path,
) -> None:
    stub = clis.Stub(
        tmp_path / "ws" / ".stub", "codex", stdout=clis.answer("codex", "done"), shebang=True
    )
    stub.install()
    task = workspace_task(
        tmp_path / "ws",
        env=stub.shell_env(),
        allowed_hosts={ANY_HOST},
        require_isolation=True,
        track_files=False,
    )

    trace = await rollout(task(), CodexCLIAgent(), runtime=LocalRuntime(task.env))

    assert "/codex is a launcher script, but a process-bound inference connection " in (
        trace.error or ""
    )
    assert stub.captures() == []


PI_WRITE_CALL = {
    "type": "toolCall",
    "id": "call-1",
    "name": "write",
    "arguments": {"path": "a.txt", "content": "x"},
}


@pytest.mark.parametrize(
    ("result", "is_error"),
    [
        (clis.pi_tool_result("call-1", "write", "Successfully wrote to a.txt"), False),
        (clis.pi_tool_result("call-1", "write", "EACCES: a.txt", is_error=True), True),
    ],
    ids=["a-tool-that-succeeded", "a-tool-that-failed"],
)
async def test_each_tool_pi_ran_lands_on_the_trace_as_a_tool_step(
    result: dict[str, Any], is_error: bool, tmp_path: Path
) -> None:
    stdout = clis.jsonl(
        clis.PI_STARTED,
        clis.pi_assistant({"type": "text", "text": "writing"}, PI_WRITE_CALL, stop="toolUse"),
        result,
        clis.pi_assistant({"type": "text", "text": "done"}),
        clis.PI_SETTLED,
    )
    stub = clis.Stub(tmp_path / "ws" / ".stub", "pi", stdout=stdout)
    stub.install()

    task = workspace_task(tmp_path / "ws", env=stub.shell_env())
    (trace,) = (await task().run(PiAgent())).traces

    (recorded,) = [step for step in trace.steps if isinstance(step, ToolStep)]
    assert recorded.call is not None and recorded.result is not None
    sent = [getattr(block, "text", None) for block in recorded.result.content]
    assert (trace.status, trace.content) == ("completed", "done")
    assert (recorded.call.name, recorded.call.arguments) == ("write", PI_WRITE_CALL["arguments"])
    assert (recorded.result.isError, sent) == (is_error, [result["message"]["content"][0]["text"]])


@pytest.mark.parametrize(
    ("stdout", "stderr", "exit", "content", "error"),
    [
        (
            clis.jsonl(
                clis.PI_STARTED,
                clis.pi_assistant(stop="error", error="400: unknown model 'no-such'"),
                clis.PI_SETTLED,
            ),
            "",
            0,
            None,
            "400: unknown model 'no-such'",
        ),
        ("", "Cannot find module", 1, None, "Cannot find module"),
        (
            clis.jsonl(clis.PI_STARTED, clis.pi_assistant({"type": "text", "text": "Fixed."})),
            "",
            0,
            "Fixed.",
            "pi CLI exited without an agent_settled event",
        ),
    ],
    ids=[
        "a-failed-model-call-with-a-zero-exit",
        "a-nonzero-exit",
        "a-zero-exit-without-settling",
    ],
)
async def test_how_pi_exits_decides_how_the_run_ends(
    stdout: str, stderr: str, exit: int, content: str | None, error: str, tmp_path: Path
) -> None:
    stub = clis.Stub(tmp_path / "ws" / ".stub", "pi", stdout=stdout, stderr=stderr, exit=exit)
    stub.install()

    task = workspace_task(tmp_path / "ws", env=stub.shell_env())
    (trace,) = (await task().run(PiAgent())).traces

    assert (trace.status, trace.content, trace.error) == (
        "error",
        content,
        f"[agent loop] RuntimeError: {error}",
    )


async def test_pi_records_each_model_call_with_its_token_usage(tmp_path: Path) -> None:
    stdout = clis.jsonl(
        clis.PI_STARTED,
        clis.pi_assistant({"type": "thinking", "thinking": "easy"}, {"type": "text", "text": "4"}),
        clis.PI_SETTLED,
    )
    stub = clis.Stub(tmp_path / "ws" / ".stub", "pi", stdout=stdout)
    stub.install()

    task = workspace_task(tmp_path / "ws", env=stub.shell_env())
    (trace,) = (await task().run(PiAgent())).traces

    (step,) = [step for step in trace.steps if isinstance(step, AgentStep)]
    assert step.usage is not None
    assert (step.content, step.reasoning, step.model) == ("4", "easy", "pi-test")
    assert (step.usage.prompt_tokens, step.usage.completion_tokens, step.usage.cached_tokens) == (
        12,
        4,
        2,
    )


async def test_pi_calls_the_environments_mcp_tools_with_their_token(tmp_path: Path) -> None:
    server = FastMCP(name="lookup")

    @server.tool
    def lookup(key: str) -> str:
        return f"value of {key}"

    seen: list[str | None] = []

    async def app(scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            seen.append(dict(scope["headers"]).get(b"authorization", b"").decode() or None)
        await inner(scope, receive, send)

    inner = server.http_app(path="/mcp")
    stub = clis.Stub(
        tmp_path / "ws" / ".stub",
        "pi",
        stdout=clis.answer("pi", "done"),
        mcp=[{"name": "lookup", "tool": "lookup", "arguments": {"key": "k"}}],
    )
    stub.install()

    async with serve_asgi(app) as port:
        task = workspace_task(
            tmp_path / "ws",
            env=stub.shell_env(),
            capabilities=(
                Capability.mcp(
                    name="lookup", url=f"http://127.0.0.1:{port}/mcp", auth_token="mcp-token"
                ),
            ),
        )
        (trace,) = (await task().run(PiAgent())).traces

    (capture,) = stub.captures()
    assert trace.status == "completed"
    assert capture["mcp"] == [["value of k"]]
    assert set(seen) == {"Bearer mcp-token"}
    assert "mcp-token" not in str(capture["argv"])


async def test_pi_without_a_model_endpoint_is_refused(hud_env: HudEnv, tmp_path: Path) -> None:
    hud_env.set(HUD_API_KEY=None)
    stub = clis.Stub(tmp_path / "ws" / ".stub", "pi", stdout=clis.answer("pi", "done"))
    stub.install()

    task = workspace_task(tmp_path / "ws", env=stub.shell_env())
    (trace,) = (await task().run(PiAgent())).traces

    assert trace.error is not None
    assert "pi needs a model endpoint" in trace.error
    assert stub.captures() == []


PI_CAPPED_RUN = clis.jsonl(
    clis.PI_STARTED,
    clis.pi_assistant({"type": "text", "text": "writing"}, PI_WRITE_CALL, stop="toolUse"),
    clis.pi_tool_result("call-1", "write", "Successfully wrote to a.txt"),
    clis.PI_TURN_END,
    clis.pi_assistant(stop="aborted", error="Request was aborted"),
    clis.PI_TURN_END,
    clis.PI_ABORTED,
)


@pytest.mark.parametrize(
    ("max_steps", "status", "stop_reason", "error"),
    [
        (1, "completed", "max_steps", None),
        (-1, "error", None, "[agent loop] RuntimeError: Request was aborted"),
    ],
    ids=["at-its-step-limit", "without-a-step-limit"],
)
async def test_a_run_pi_aborts_ends_by_whether_it_reached_max_steps(
    max_steps: int, status: str, stop_reason: str | None, error: str | None, tmp_path: Path
) -> None:
    stub = clis.Stub(tmp_path / "ws" / ".stub", "pi", stdout=PI_CAPPED_RUN)
    stub.install()

    task = workspace_task(tmp_path / "ws", env=stub.shell_env())
    (trace,) = (await task().run(PiAgent(PiConfig(max_steps=max_steps)))).traces

    assert (trace.status, trace.stop_reason, trace.error, trace.content) == (
        status,
        stop_reason,
        error,
        "writing",
    )


async def test_a_pi_answer_cut_off_at_the_output_limit_stops_for_length(tmp_path: Path) -> None:
    stdout = clis.jsonl(
        clis.PI_STARTED,
        clis.pi_assistant({"type": "text", "text": "par"}, stop="length"),
        clis.PI_SETTLED,
    )
    stub = clis.Stub(tmp_path / "ws" / ".stub", "pi", stdout=stdout)
    stub.install()

    task = workspace_task(tmp_path / "ws", env=stub.shell_env())
    (trace,) = (await task().run(PiAgent())).traces

    assert (trace.status, trace.stop_reason, trace.content) == ("completed", "length", "par")


@pytest.mark.parametrize(
    ("fields", "named"),
    [
        ({"completion_kwargs": {"extra_body": {"return_token_ids": True}}}, "token ids"),
        ({"extensions": {"../escape.ts": ""}}, "extensions"),
    ],
    ids=["token-ids", "an-extension-path"],
)
def test_pi_config_refuses_what_pi_cannot_honor(fields: dict[str, Any], named: str) -> None:
    with pytest.raises(ValidationError, match=named):
        PiConfig.model_validate(fields)


ECHO_EXTENSION = """
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "echo_word",
    label: "Echo word",
    description: "Return the given word unchanged.",
    parameters: Type.Object({ word: Type.String() }),
    async execute(_id, params) {
      return { content: [{ type: "text", text: params.word }], details: {} };
    },
  });
  pi.on("tool_call", (event) => {
    if (event.toolName === "echo_word") event.input.word = "pineapple";
  });
}
"""


@pytest.mark.e2e
@pytest.mark.live
async def test_a_configured_pi_extension_adds_its_tool_and_hooks_its_calls(
    live: HudEnv, tmp_path: Path
) -> None:
    """The model asks to echo banana; only the extension's hook can make the answer pineapple."""
    env = Environment("ext")
    env.workspace(tmp_path, guest_path=str(tmp_path))

    @env.template()
    async def echo():
        answer = yield (
            "Call the echo_word tool with the word banana, then reply with exactly the word "
            "it returned."
        )
        yield 1.0 if "pineapple" in answer.lower() else 0.0

    agent = PiAgent(
        PiConfig(
            model="deepseek/deepseek-v4.1-flash",
            extensions={"echo.ts": ECHO_EXTENSION},
            max_steps=4,
            timeout_seconds=120,
        )
    )
    (trace,) = (await echo().run(agent)).traces

    assert (trace.reward, trace.error) == (1.0, None), f"answered {trace.content!r}"
