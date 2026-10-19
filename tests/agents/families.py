"""The agent families every agent contract runs over, and how a scenario drives each one.

A family is one agent implementation in ``hud.agents``. A scenario is written once
against a :class:`Driver`, which builds the family's agent, tells its model what to
do (a scripted provider, or a stub CLI on the workspace's ``PATH``), runs it on a
workspace, and reports what the model's side saw. A contract's table maps every
family to its row or to a :class:`NotApplicable` reason; :func:`over_families`
fails collection when a family is missing, so a new agent without a row fails the
suite. ``NOT_FAMILIES`` names the agents in ``hud.agents`` that no contract here
covers, with the reason.
"""

from __future__ import annotations

import json
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol

import pytest
from fastmcp import FastMCP

from hud.agents import (
    ClaudeAgent,
    ClaudeCLIAgent,
    CodexCLIAgent,
    GeminiAgent,
    OpenAIAgent,
    OpenAIChatAgent,
)
from hud.agents.browser_use import BrowserUseAgent
from hud.agents.types import (
    BrowserUseConfig,
    ClaudeCLIConfig,
    ClaudeConfig,
    CodexCLIConfig,
    GeminiConfig,
    OpenAIChatConfig,
    OpenAIConfig,
)
from hud.capabilities import Capability
from tests.agents.cli_stub import Stub, jsonl
from tests.support import (
    Reply,
    Turn,
    call,
    mcp_server,
    run_task,
    say,
    shell_call,
    workspace_env,
)
from tests.support.envs import PROMPT

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Mapping
    from contextlib import AbstractAsyncContextManager
    from pathlib import Path

    from _pytest.mark import ParameterSet

    from hud import Environment
    from hud.agents.base import Agent
    from hud.eval import Run
    from tests.support import FakeBrowser, ModelRequest, Models

#: How long a hung provider or CLI waits; any agent timeout a test sets is far shorter.
HANG_SECONDS = 30.0


@dataclass(frozen=True)
class NotApplicable:
    """A table entry for a family the contract does not cover, and why."""

    reason: str


def over_families(table: Mapping[str, object]) -> list[ParameterSet]:
    """``(family, row)`` parameters for every family the table covers.

    A table must name every family, as a row or as :class:`NotApplicable`.
    """
    missing = FAMILIES.keys() - table.keys()
    unknown = table.keys() - FAMILIES.keys()
    assert not missing and not unknown, f"families without an entry {missing}, unknown {unknown}"
    return [
        pytest.param(name, row, id=name)
        for name, row in table.items()
        if not isinstance(row, NotApplicable)
    ]


def over_family_cases(
    table: Mapping[str, Mapping[str, object] | NotApplicable],
) -> list[ParameterSet]:
    """``(family, case)`` parameters for a table whose rows hold several named cases."""
    over_families(table)
    return [
        pytest.param(name, case, id=f"{name}-{case_id}")
        for name, cases in table.items()
        if not isinstance(cases, NotApplicable)
        for case_id, case in cases.items()
    ]


@dataclass(frozen=True)
class Seen:
    """One model request, or one CLI invocation, as the model's side received it.

    ``route`` is ``"gateway"`` (the HUD gateway), ``"provider"`` (the provider's own
    API) or ``"ambient"`` (a CLI left to its own login). ``budget`` is the turn
    budget handed to a CLI that runs its own loop.
    """

    prompt: str
    tools: frozenset[str]
    route: str
    credential: str | None
    trace_id: str | None
    model: str
    budget: int | None = None


class Driver(Protocol):
    """How a scenario drives one family. Get one from the ``driver`` fixture."""

    @property
    def family(self) -> Family:
        """The family this driver drives."""
        ...

    @property
    def workspace(self) -> Path:
        """The workspace directory the agent's tools act on."""
        ...

    def agent(self, **config: Any) -> Agent:
        """The family's agent, with ``config`` over its defaults."""
        ...

    def answer(self, text: str) -> None:
        """The model replies ``text`` and the run ends."""
        ...

    def shell(self, command: str, *, then: str) -> None:
        """The model runs ``command`` in the workspace, then replies ``then``."""
        ...

    def persist(self) -> None:
        """The model never finishes on its own."""
        ...

    def hang(self) -> None:
        """The model's reply never comes."""
        ...

    def fault(self, message: str) -> None:
        """The provider fails the request with ``message``."""
        ...

    def environment(self, capabilities: tuple[Capability, ...] = ()) -> Environment:
        """The workspace environment the family runs in."""
        ...

    async def run(
        self, agent: Agent, *, prompt: str = PROMPT, capabilities: tuple[Capability, ...] = ()
    ) -> Run:
        """Roll ``agent`` out on a fresh workspace environment."""
        ...

    def seen(self) -> list[Seen]:
        """Every model request or CLI invocation so far."""
        ...

    def processes(self) -> list[int]:
        """Process ids of every CLI the agent launched."""
        ...


# ─── provider tool-call agents ─────────────────────────────────────────


@dataclass(frozen=True)
class ToolFamily:
    """An agent that runs hud's tool loop against a provider API."""

    name: str
    agent_cls: type[Agent]
    config: Callable[..., Any]
    #: The model's call running a command in the workspace shell.
    shell: Callable[[str], Turn]
    #: The tools a workspace gives the model.
    workspace_tools: frozenset[str]
    key: str
    header: str


def responses_tools(request: ModelRequest) -> frozenset[str]:
    """Responses API tools by name, or by type for a provider-native tool such as ``shell``."""
    return frozenset(tool.get("name") or tool["type"] for tool in request.body.get("tools") or [])


@dataclass
class ToolDriver:
    family: ToolFamily
    models: Models
    provider: Models
    root: Path

    @property
    def workspace(self) -> Path:
        return self.root

    def agent(self, **config: Any) -> Agent:
        return self.family.agent_cls(self.family.config(**config))

    def script(self, turns: list[Turn]) -> None:
        self.models.script(turns)
        self.provider.script(turns)

    def respond(self, policy: Callable[[ModelRequest], Turn]) -> None:
        self.models.respond(policy)
        self.provider.respond(policy)

    def answer(self, text: str) -> None:
        self.script([say(text)])

    def shell(self, command: str, *, then: str) -> None:
        self.script([self.family.shell(command), say(then)])

    def persist(self) -> None:
        self.respond(lambda _request: self.family.shell("true"))

    def hang(self) -> None:
        self.script([Turn(fail=Reply(status=500, delay=HANG_SECONDS))])

    def fault(self, message: str) -> None:
        self.script([Turn(fail=Reply(status=400, json={"error": {"message": message}}))])

    def environment(self, capabilities: tuple[Capability, ...] = ()) -> Environment:
        return workspace_env(self.root, capabilities=capabilities)

    async def run(
        self, agent: Agent, *, prompt: str = PROMPT, capabilities: tuple[Capability, ...] = ()
    ) -> Run:
        return await run_task(self.environment(capabilities), agent, prompt=prompt)

    def seen(self) -> list[Seen]:
        return [
            Seen(
                prompt=request.prompt,
                tools=responses_tools(request)
                if request.protocol == "responses"
                else frozenset(request.tools),
                route=route,
                credential=request.headers.get(self.family.header, "").removeprefix("Bearer "),
                trace_id=request.headers.get("trace-id"),
                model=request.model,
            )
            for route, server in (("gateway", self.models), ("provider", self.provider))
            for request in server.requests()
        ]

    def processes(self) -> list[int]:
        return []


# ─── CLI agents ────────────────────────────────────────────────────────


def claude_assistant(*content: dict[str, Any], stop: str = "end_turn") -> dict[str, Any]:
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


def claude_result(text: str = "done", *, is_error: bool = False) -> dict[str, Any]:
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


def claude_answer(text: str) -> str:
    return jsonl(claude_assistant({"type": "text", "text": text}), claude_result(text))


def claude_shell(command: str, output: str, then: str) -> str:
    return jsonl(
        claude_assistant(
            {"type": "tool_use", "id": "tool-1", "name": "Bash", "input": {"command": command}},
            stop="tool_use",
        ),
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tool-1", "content": output}],
            },
        },
        claude_assistant({"type": "text", "text": then}),
        claude_result(then),
    )


def claude_fault(message: str) -> str:
    return jsonl(claude_result(message, is_error=True))


def codex_item(kind: str, item_id: str, **fields: Any) -> dict[str, Any]:
    return {"type": "item.completed", "item": {"id": item_id, "type": kind, **fields}}


CODEX_STARTED = {"type": "thread.started", "thread_id": "thread-1"}
CODEX_COMPLETED = {
    "type": "turn.completed",
    "usage": {"input_tokens": 20, "cached_input_tokens": 5, "output_tokens": 8},
}


def codex_answer(text: str) -> str:
    return jsonl(
        CODEX_STARTED, codex_item("agent_message", "message-1", text=text), CODEX_COMPLETED
    )


def codex_shell(command: str, output: str, then: str) -> str:
    return jsonl(
        CODEX_STARTED,
        codex_item(
            "command_execution",
            "command-1",
            command=command,
            aggregated_output=output,
            exit_code=0,
            status="completed",
        ),
        codex_item("agent_message", "message-1", text=then),
        CODEX_COMPLETED,
    )


def codex_fault(message: str) -> str:
    return jsonl({"type": "turn.started"}, {"type": "turn.failed", "error": {"message": message}})


def overrides(argv: list[str]) -> dict[str, str]:
    """The ``-c key=value`` overrides in a Codex command line."""
    return dict(argv[index + 1].split("=", 1) for index, arg in enumerate(argv) if arg == "-c")


def flag(argv: list[str], name: str) -> str | None:
    """The value of ``name value`` or ``name=value`` in a command line."""
    for index, arg in enumerate(argv):
        if arg == name:
            return argv[index + 1]
        if arg.startswith(f"{name}="):
            return arg.removeprefix(f"{name}=")
    return None


def claude_seen(capture: dict[str, Any], gateway: str) -> Seen:
    env, argv = capture["env"], capture["argv"]
    (message,) = [json.loads(line) for line in capture["stdin"].splitlines()]
    servers = json.loads(capture["files"].get(".hud_mcp_config.json", '{"mcpServers": {}}'))
    allowed = {argv[index + 1] for index, arg in enumerate(argv) if arg == "--allowedTools"}
    headers = dict(
        line.split(": ", 1) for line in env.get("ANTHROPIC_CUSTOM_HEADERS", "").splitlines()
    )
    budget = flag(argv, "--max-turns")
    if env.get("ANTHROPIC_BASE_URL") == gateway:
        route = "gateway"
    else:
        route = "provider" if "ANTHROPIC_API_KEY" in env else "ambient"
    return Seen(
        prompt=message["message"]["content"][0]["text"],
        tools=frozenset(allowed | servers["mcpServers"].keys()),
        route=route,
        credential=env.get("ANTHROPIC_API_KEY"),
        trace_id=headers.get("Trace-Id"),
        model=env["ANTHROPIC_MODEL"],
        budget=None if budget is None else int(budget),
    )


def codex_seen(capture: dict[str, Any], gateway: str) -> Seen:
    env, argv = capture["env"], capture["argv"]
    settings = overrides(argv)
    base_url = settings.get("model_providers.hud.base_url")
    trace = re.search(r'"Trace-Id"="([^"]+)"', settings.get("model_providers.hud.http_headers", ""))
    if base_url is not None and json.loads(base_url) == gateway:
        route, credential = "gateway", env.get(json.loads(settings["model_providers.hud.env_key"]))
    else:
        route = "provider" if "CODEX_API_KEY" in env else "ambient"
        credential = env.get("CODEX_API_KEY")
    model = flag(argv, "--model")
    assert model is not None
    return Seen(
        prompt=capture["stdin"],
        tools=frozenset(),
        route=route,
        credential=credential,
        trace_id=trace and trace.group(1),
        model=model,
    )


@dataclass(frozen=True)
class CliFamily:
    """An agent that runs a coding CLI in the workspace and records its JSONL stream."""

    name: str
    agent_cls: type[Agent]
    config: Callable[..., Any]
    executable: str
    answer: Callable[[str], str]
    shell: Callable[[str, str, str], str]
    fault: Callable[[str], str]
    seen: Callable[[dict[str, Any], str], Seen]
    key: str


@dataclass
class CliDriver:
    family: CliFamily
    models: Models
    root: Path
    stub: Stub = field(init=False)

    def __post_init__(self) -> None:
        self.stub = Stub(self.root / "stub", self.family.executable)

    @property
    def workspace(self) -> Path:
        return self.root / "ws"

    def agent(self, **config: Any) -> Agent:
        return self.family.agent_cls(self.family.config(**config))

    def behave(self, **behavior: Any) -> None:
        """Reinstall the stub with ``behavior`` (``stdout``, ``exit``, ``hang``...)."""
        self.stub = replace(Stub(self.stub.root, self.stub.name), **behavior)
        self.stub.install()

    def answer(self, text: str) -> None:
        self.behave(stdout=self.family.answer(text))

    def shell(self, command: str, *, then: str) -> None:
        self.behave(commands=[command], stdout=self.family.shell(command, "ran", then))

    def persist(self) -> None:
        self.answer("done")

    def hang(self) -> None:
        self.behave(stdout=self.family.answer("done").splitlines()[0] + "\n", hang=True)

    def fault(self, message: str) -> None:
        self.behave(stdout=self.family.fault(message))

    def environment(self, capabilities: tuple[Capability, ...] = ()) -> Environment:
        return workspace_env(self.workspace, env=self.stub.shell_env(), capabilities=capabilities)

    async def run(
        self, agent: Agent, *, prompt: str = PROMPT, capabilities: tuple[Capability, ...] = ()
    ) -> Run:
        return await run_task(self.environment(capabilities), agent, prompt=prompt)

    def seen(self) -> list[Seen]:
        return [self.family.seen(capture, self.models.url) for capture in self.stub.captures()]

    def processes(self) -> list[int]:
        return [capture["pid"] for capture in self.stub.captures()]


# ─── browser-use ───────────────────────────────────────────────────────


@dataclass
class BrowserUseDriver:
    """Drives ``BrowserUseAgent`` through the stand-in browser-use SDK and a fake browser.

    browser-use asks the model through its Anthropic client, so the model's side is
    the scripted provider (the agent never routes through the HUD gateway); it
    finishes when the model replies ``done: <answer>``.
    """

    family: BrowserUseFamily
    models: Models
    provider: Models
    root: Path
    browser: FakeBrowser

    @property
    def workspace(self) -> Path:
        return self.root

    def agent(self, **config: Any) -> Agent:
        return self.family.agent_cls(self.family.config(**config))

    def script(self, turns: list[Turn]) -> None:
        self.models.script(turns)
        self.provider.script(turns)

    def answer(self, text: str) -> None:
        self.script([say(f"done: {text}")])

    def shell(self, command: str, *, then: str) -> None:
        raise NotImplementedError("browser-use acts on the browser, not the workspace shell")

    def persist(self) -> None:
        self.models.respond(lambda _request: say("still looking"))
        self.provider.respond(lambda _request: say("still looking"))

    def hang(self) -> None:
        self.script([Turn(fail=Reply(status=500, delay=HANG_SECONDS))])

    def fault(self, message: str) -> None:
        self.script([Turn(fail=Reply(status=400, json={"error": {"message": message}}))])

    def environment(self, capabilities: tuple[Capability, ...] = ()) -> Environment:
        return workspace_env(
            self.root, capabilities=(Capability.cdp(url=self.browser.url), *capabilities)
        )

    async def run(
        self, agent: Agent, *, prompt: str = PROMPT, capabilities: tuple[Capability, ...] = ()
    ) -> Run:
        return await run_task(self.environment(capabilities), agent, prompt=prompt)

    def seen(self) -> list[Seen]:
        return [
            Seen(
                prompt=request.prompt,
                tools=frozenset(request.tools),
                route=route,
                credential=request.headers.get("x-api-key"),
                trace_id=request.headers.get("trace-id"),
                model=request.model,
            )
            for route, server in (("gateway", self.models), ("provider", self.provider))
            for request in server.requests()
        ]

    def processes(self) -> list[int]:
        return []


@dataclass(frozen=True)
class BrowserUseFamily:
    name: str = "browser_use"
    agent_cls: type[Agent] = BrowserUseAgent
    key: str = "ANTHROPIC_API_KEY"
    config: Callable[..., Any] = BrowserUseConfig


# ─── the families ──────────────────────────────────────────────────────


Family = ToolFamily | CliFamily | BrowserUseFamily

CLAUDE = ToolFamily(
    name="claude",
    agent_cls=ClaudeAgent,
    config=lambda **config: ClaudeConfig.model_validate({"model": "claude-sonnet-4-6", **config}),
    shell=lambda command: call("bash", command=command),
    workspace_tools=frozenset({"bash", "str_replace_based_edit_tool"}),
    key="ANTHROPIC_API_KEY",
    header="x-api-key",
)
OPENAI = ToolFamily(
    name="openai",
    agent_cls=OpenAIAgent,
    config=lambda **config: OpenAIConfig.model_validate({"model": "gpt-5.6", **config}),
    shell=lambda command: shell_call(command),
    workspace_tools=frozenset({"shell"}),
    key="OPENAI_API_KEY",
    header="authorization",
)
GEMINI = ToolFamily(
    name="gemini",
    agent_cls=GeminiAgent,
    config=lambda **config: GeminiConfig.model_validate(
        {"model": "gemini-3.1-pro-preview", **config}
    ),
    shell=lambda command: call("run_shell_command", command=command),
    workspace_tools=frozenset(
        {
            "run_shell_command",
            "replace",
            "write_file",
            "read_file",
            "grep_search",
            "glob",
            "list_directory",
        }
    ),
    key="GEMINI_API_KEY",
    header="x-goog-api-key",
)
CHAT = ToolFamily(
    name="openai_compatible",
    agent_cls=OpenAIChatAgent,
    config=lambda **config: OpenAIChatConfig.model_validate({"model": "qwen3.6-plus", **config}),
    shell=lambda command: call("bash", command=command),
    workspace_tools=frozenset({"bash", "read", "glob", "grep", "edit", "write"}),
    key="OPENAI_API_KEY",
    header="authorization",
)
CLAUDE_CLI = CliFamily(
    name="claude_cli",
    agent_cls=ClaudeCLIAgent,
    config=lambda **config: ClaudeCLIConfig(**config),
    executable="claude",
    answer=claude_answer,
    shell=claude_shell,
    fault=claude_fault,
    seen=claude_seen,
    key="ANTHROPIC_API_KEY",
)
CODEX_CLI = CliFamily(
    name="codex_cli",
    agent_cls=CodexCLIAgent,
    config=lambda **config: CodexCLIConfig(**config),
    executable="codex",
    answer=codex_answer,
    shell=codex_shell,
    fault=codex_fault,
    seen=codex_seen,
    key="OPENAI_API_KEY",
)
BROWSER_USE = BrowserUseFamily()

TOOL_FAMILIES: dict[str, ToolFamily] = {
    family.name: family for family in (CLAUDE, OPENAI, GEMINI, CHAT)
}
CLI_FAMILIES: dict[str, CliFamily] = {family.name: family for family in (CLAUDE_CLI, CODEX_CLI)}
FAMILIES: dict[str, Family] = {**TOOL_FAMILIES, **CLI_FAMILIES, BROWSER_USE.name: BROWSER_USE}

#: Agents in ``hud.agents`` outside these contracts.
NOT_FAMILIES = {
    "RobotAgent": "drives a robot capability with a policy model and no model provider; "
    "tests/robot holds its contracts",
    "BatchedAgent": "a RobotAgent that batches policy forwards; tests/robot",
}


# ─── wire translation ──────────────────────────────────────────────────


@asynccontextmanager
async def no_capabilities() -> AsyncIterator[tuple[Capability, ...]]:
    yield ()


@dataclass(frozen=True)
class Wire:
    """A configuration a provider agent must carry onto its requests.

    The agent runs once with ``config`` on a workspace holding ``files`` and the
    capabilities ``serve`` yields, against a model scripted with ``turns``;
    ``read`` picks the part of the requests the configuration decides.
    """

    config: dict[str, Any]
    read: Callable[[list[ModelRequest]], Any]
    expected: Any
    turns: list[Turn] = field(default_factory=lambda: [say("done")])
    serve: Callable[[], AbstractAsyncContextManager[tuple[Capability, ...]]] = no_capabilities
    files: dict[str, bytes | str] = field(default_factory=dict[str, bytes | str])


async def carried(family: ToolFamily, wire: Wire, models: Models, root: Path) -> Any:
    """Run ``family`` as ``wire`` configures it and read what reached the provider."""
    models.script(wire.turns)
    async with wire.serve() as capabilities:
        env = workspace_env(root, files=wire.files, capabilities=capabilities)
        run = await run_task(env, family.agent_cls(family.config(**wire.config)))
    assert run.trace.status == "completed", run.trace.error
    return wire.read(models.requests())


def lookups() -> FastMCP:
    server = FastMCP("lookups")
    server.tool(name="lookup", description="Look a key up.")(lambda key: key)
    return server


@asynccontextmanager
async def two_servers() -> AsyncIterator[tuple[Capability, ...]]:
    """MCP servers ``database`` and ``search``, each with a ``lookup`` tool."""
    async with (
        mcp_server(lookups(), name="database") as database,
        mcp_server(lookups(), name="search") as search,
    ):
        yield (database, search)
