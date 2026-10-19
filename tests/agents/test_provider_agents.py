"""What every provider agent implements over its own API: what its model is handed, how
its replies and tool calls are read, how its provider's failures end or retry a run,
and where its requests go with which credential. The tool loop they share is
``test_tool_loop``; what one provider alone sends is in its own module."""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest
from fastmcp import FastMCP

from hud.agents import ClaudeAgent, GeminiAgent, OpenAIAgent, OpenAIChatAgent
from hud.agents.types import (
    ClaudeConfig,
    GeminiConfig,
    OpenAIChatConfig,
    OpenAIConfig,
    ToolStep,
)
from hud.eval import LocalRuntime, Task, Taskset
from hud.utils.exceptions import HudAuthenticationError
from tests.fixtures.environments import run_task, workspace_env
from tests.support import Reply, Turn, call, mcp_server, say, shell_call, stream_error

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from hud.agents.base import Agent
    from tests.support import HudEnv, ModelRequest, Models


@dataclass(frozen=True)
class Provider:
    """An agent that runs hud's tool loop against its provider's API."""

    name: str
    agent: Callable[..., Agent]
    #: The model's call running a command in the workspace shell.
    shell: Callable[[str], Turn]
    #: The tools a workspace gives the model.
    workspace_tools: frozenset[str]
    #: The provider's own key, and the header a request carries it in.
    key: str
    header: str

    def tools(self, request: ModelRequest) -> frozenset[str]:
        """The tools a request offers, by name (by type for a provider-native tool)."""
        if request.protocol == "responses":
            return frozenset(tool.get("name") or tool["type"] for tool in request.body["tools"])
        return frozenset(request.tools)

    def credential(self, request: ModelRequest) -> str:
        return request.headers.get(self.header, "").removeprefix("Bearer ")


CLAUDE = Provider(
    "claude",
    lambda **config: ClaudeAgent(
        ClaudeConfig.model_validate({"model": "claude-sonnet-4-6", **config})
    ),
    lambda command: call("bash", command=command),
    frozenset({"bash", "str_replace_based_edit_tool"}),
    "ANTHROPIC_API_KEY",
    "x-api-key",
)
OPENAI = Provider(
    "openai",
    lambda **config: OpenAIAgent(OpenAIConfig.model_validate({"model": "gpt-5.6", **config})),
    shell_call,
    frozenset({"shell"}),
    "OPENAI_API_KEY",
    "authorization",
)
GEMINI = Provider(
    "gemini",
    lambda **config: GeminiAgent(
        GeminiConfig.model_validate({"model": "gemini-3.1-pro-preview", **config})
    ),
    lambda command: call("run_shell_command", command=command),
    frozenset(
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
    "GEMINI_API_KEY",
    "x-goog-api-key",
)
CHAT = Provider(
    "openai_compatible",
    lambda **config: OpenAIChatAgent(
        OpenAIChatConfig.model_validate({"model": "qwen3.6-plus", **config})
    ),
    lambda command: call("bash", command=command),
    frozenset({"bash", "read", "glob", "grep", "edit", "write"}),
    "OPENAI_API_KEY",
    "authorization",
)
PROVIDERS = {family.name: family for family in (CLAUDE, OPENAI, GEMINI, CHAT)}

EVERY_PROVIDER = pytest.mark.parametrize("family", PROVIDERS.values(), ids=PROVIDERS.keys())


@pytest.fixture(autouse=True)
def signed_in(hud_env: HudEnv) -> None:
    hud_env.set(HUD_API_KEY="hud-key")


def lookup_server() -> FastMCP:
    server = FastMCP("tools")
    server.tool(name="lookup", description="Look a key up.")(lambda key: key)
    return server


@EVERY_PROVIDER
async def test_the_model_is_given_the_prompt_and_the_environments_tools(
    family: Provider, models: Models, tmp_path: Path
) -> None:
    models.script([say("done")])

    async with mcp_server(lookup_server()) as server:
        env = workspace_env(tmp_path / "ws", capabilities=(server,))
        await run_task(env, family.agent(), prompt="Fix the bug.")

    (request,) = models.requests()
    assert (request.prompt, family.tools(request)) == (
        "Fix the bug.",
        family.workspace_tools | {"lookup"},
    )


@EVERY_PROVIDER
async def test_one_agent_drives_concurrent_rollouts_each_with_its_own_conversation(
    family: Provider, models: Models, tmp_path: Path
) -> None:
    models.script([say("done")])
    env, agent = workspace_env(tmp_path / "ws"), family.agent()

    runs = await asyncio.gather(
        run_task(env, agent, prompt="first"), run_task(env, agent, prompt="second")
    )

    assert [run.trace.status for run in runs] == ["completed", "completed"]
    assert sorted(request.prompt for request in models.requests()) == ["first", "second"]


# ─── its tools ─────────────────────────────────────────────────────────

SHELL_STEP = {"claude": "bash", "openai": "shell", "gemini": "run_shell_command"}


@EVERY_PROVIDER
async def test_a_shell_call_acts_on_the_workspace_and_lands_on_the_trace(
    family: Provider, models: Models, tmp_path: Path
) -> None:
    models.script([family.shell("touch made"), say("done")])

    run = await run_task(workspace_env(tmp_path / "ws"), family.agent())

    calls = [step.call.name for step in run.trace.steps if isinstance(step, ToolStep) and step.call]
    assert (run.trace.status, run.trace.content) == ("completed", "done")
    assert ((tmp_path / "ws" / "made").exists(), calls) == (
        True,
        [SHELL_STEP.get(family.name, "bash")],
    )


# A call the provider's own tool refuses, and what the model is told.
REFUSED_CALLS: dict[str, tuple[Turn, str]] = {
    "claude": (
        call("str_replace_based_edit_tool", command="bogus", path="/x"),
        "unknown editor command: 'bogus'",
    ),
    "openai": (
        Turn(native=({**OPENAI.shell("true").native[0], "action": {"commands": 5}},)),
        "commands must be a list of strings",
    ),
    "gemini": (call("run_shell_command", command=""), "tool error: command is required"),
    "openai_compatible": (call("bash", command=""), "tool error: command is required"),
}


@EVERY_PROVIDER
async def test_a_call_its_own_tool_refuses_is_told_to_the_model_and_the_run_goes_on(
    family: Provider, models: Models, tmp_path: Path
) -> None:
    turn, told = REFUSED_CALLS[family.name]
    models.script([turn, say("done")])

    run = await run_task(workspace_env(tmp_path / "ws"), family.agent())

    assert (run.trace.status, run.trace.content) == ("completed", "done")
    assert told in json.dumps(models.requests()[-1].body)


CUT_OFF = {
    "openai": Turn(
        native=(
            {
                "type": "function_call",
                "id": "fc_cut",
                "call_id": "call_cut",
                "name": "bash",
                "arguments": '{"command": "ec',
                "status": "completed",
            },
        )
    ),
    "openai_compatible": Turn(
        native=(
            {
                "id": "call_cut",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"command": "ec'},
            },
        )
    ),
}
BAD_JSON = Turn(
    native=({"type": "tool_use", "id": "toolu_bad", "name": "bash", "input": '{"command": ls}'},)
)
INCOMPLETE = "the arguments for this 'bash' call arrived incomplete"


@pytest.mark.parametrize(
    ("family", "turns", "told"),
    [
        pytest.param(
            CLAUDE,
            [BAD_JSON, BAD_JSON, say("done")],
            "Your previous tool-call arguments were invalid JSON.",
            id="claude",
        ),
        pytest.param(OPENAI, [CUT_OFF["openai"], say("done")], INCOMPLETE, id="openai"),
        pytest.param(
            CHAT, [CUT_OFF["openai_compatible"], say("done")], INCOMPLETE, id="openai_compatible"
        ),
    ],
)
async def test_a_call_with_malformed_arguments_is_explained_to_the_model(
    family: Provider, turns: list[Turn], told: str, models: Models, tmp_path: Path
) -> None:
    """Gemini reports a malformed call as its finish reason, a row of the retries below."""
    models.queue(turns)

    run = await run_task(workspace_env(tmp_path / "ws"), family.agent())

    assert (run.trace.status, run.trace.content) == ("completed", "done")
    assert told in json.dumps(models.requests()[-1].body)


@EVERY_PROVIDER
async def test_a_reply_cut_at_the_token_cap_ends_the_run_as_truncated(
    family: Provider, models: Models, tmp_path: Path
) -> None:
    models.script([Turn(text="part", truncated=True)])

    run = await run_task(workspace_env(tmp_path / "ws"), family.agent())

    trace = run.trace
    assert (trace.status, trace.stop_reason, trace.is_truncated) == ("completed", "length", True)


# ─── when its provider fails ───────────────────────────────────────────


@EVERY_PROVIDER
async def test_a_provider_fault_ends_the_run_with_its_error(
    family: Provider, models: Models, tmp_path: Path
) -> None:
    models.script([Turn(fail=Reply(status=400, json={"error": {"message": "model unavailable"}}))])

    run = await run_task(workspace_env(tmp_path / "ws"), family.agent())

    assert run.trace.status == "error"
    assert "model unavailable" in (run.trace.error or "")


TRANSIENT = {"retry-after-ms": "0"}
BUSY = Turn(fail=Reply(status=529, headers=TRANSIENT, json={"error": {"message": "busy"}}))


@pytest.mark.parametrize(
    ("family", "turns", "status", "requests", "error"),
    [
        pytest.param(CLAUDE, [BUSY, say("ok")], "completed", 2, None, id="claude-overloaded"),
        pytest.param(OPENAI, [BUSY, say("ok")], "completed", 2, None, id="openai-overloaded"),
        pytest.param(CHAT, [BUSY, say("ok")], "completed", 2, None, id="chat-overloaded"),
        pytest.param(
            CLAUDE,
            [stream_error("overloaded_error", headers=TRANSIENT), say("ok")],
            "completed",
            2,
            None,
            id="claude-a-transient-stream-error",
        ),
        pytest.param(
            CLAUDE,
            [stream_error("gateway_timeout", headers=TRANSIENT)] * 3,
            "error",
            3,
            None,
            id="claude-three-transient-stream-errors",
        ),
        pytest.param(
            CLAUDE,
            [stream_error("invalid_request_error", headers=TRANSIENT)],
            "error",
            1,
            None,
            id="claude-a-stream-request-error-is-not-retried",
        ),
        pytest.param(
            GEMINI,
            [Turn(fail=Reply(json={"candidates": []}))],
            "error",
            1,
            "Gemini returned no candidates for model gemini-3.1-pro-preview",
            id="gemini-no-candidates",
        ),
        pytest.param(
            GEMINI,
            [Turn(finish="MALFORMED_FUNCTION_CALL")],
            "error",
            1,
            "Provider returned a malformed function call",
            id="gemini-a-malformed-function-call",
        ),
    ],
)
async def test_what_a_provider_reply_lets_the_agent_retry(
    family: Provider,
    turns: list[Turn],
    status: str,
    requests: int,
    error: str | None,
    models: Models,
    tmp_path: Path,
) -> None:
    models.queue(turns)

    run = await run_task(workspace_env(tmp_path / "ws"), family.agent())

    assert (run.trace.status, len(models.requests())) == (status, requests)
    assert (error or "") in (run.trace.error or "")


# ─── where its requests go ─────────────────────────────────────────────


@dataclass(frozen=True)
class Signed:
    """How a user signs in (``None`` unsets a variable) and configures the agent, and the
    endpoint (``gateway`` or ``provider``) and credential its requests carry."""

    variables: dict[str, str | None]
    route: str
    credential: str
    config: dict[str, Any] = field(default_factory=dict[str, Any])


#: A config value standing for the provider's URL.
PROVIDER_URL = "<provider>"


def own_key_or_gateway(key: str) -> dict[str, Signed]:
    return {
        "own-key": Signed({key: "own-key"}, "provider", "own-key"),
        "hud-key-only": Signed({key: None}, "gateway", "hud-key"),
        "gateway-forced-over-own-key": Signed(
            {key: "own-key"}, "gateway", "hud-key", {"gateway": True}
        ),
    }


ROUTES: dict[str, dict[str, Signed]] = {
    "claude": own_key_or_gateway("ANTHROPIC_API_KEY"),
    "openai": own_key_or_gateway("OPENAI_API_KEY"),
    "gemini": own_key_or_gateway("GEMINI_API_KEY"),
    "openai_compatible": {
        "explicit-endpoint": Signed(
            {}, "provider", "explicit", {"api_key": "explicit", "base_url": PROVIDER_URL}
        ),
        "hud-key-only": Signed({}, "gateway", "hud-key"),
        "an-openai-key-is-not-its-own": Signed(
            {"OPENAI_API_KEY": "openai-key"}, "gateway", "hud-key"
        ),
    },
}


@pytest.mark.parametrize(
    ("family", "signed"),
    [
        pytest.param(PROVIDERS[name], signed, id=f"{name}-{case}")
        for name, cases in ROUTES.items()
        for case, signed in cases.items()
    ],
)
async def test_requests_go_where_the_sign_in_routes_them_with_its_credential(
    family: Provider,
    signed: Signed,
    models: Models,
    provider: Models,
    hud_env: HudEnv,
    tmp_path: Path,
) -> None:
    hud_env.set(
        OPENAI_BASE_URL=provider.url,
        ANTHROPIC_BASE_URL=provider.url,
        GOOGLE_GEMINI_BASE_URL=provider.url,
        **signed.variables,
    )
    config = {k: provider.url if v == PROVIDER_URL else v for k, v in signed.config.items()}
    models.script([say("done")])
    provider.script([say("done")])

    run = await run_task(workspace_env(tmp_path / "ws"), family.agent(**config))

    reached = {"gateway": models.requests(), "provider": provider.requests()}
    (request,) = reached[signed.route]
    assert run.trace.status == "completed"
    assert family.credential(request) == signed.credential
    assert ("trace-id" in request.headers) is (signed.route == "gateway")


BEDROCK = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/anthropic.claude"
AWS = {
    "AWS_ACCESS_KEY_ID": "AKIATEST",
    "AWS_SECRET_ACCESS_KEY": "secret",
    "AWS_REGION": "us-east-1",
}
NO_KEY = {"HUD_API_KEY": None}


@pytest.mark.parametrize(
    ("family", "variables", "config", "error", "message"),
    [
        pytest.param(
            CLAUDE,
            NO_KEY,
            {},
            HudAuthenticationError,
            "No API key for anthropic",
            id="claude-no-key",
        ),
        pytest.param(
            CLAUDE,
            {},
            {"model": BEDROCK},
            HudAuthenticationError,
            "AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION are required",
            id="claude-bedrock-without-aws-credentials",
        ),
        pytest.param(
            CLAUDE,
            AWS,
            {"model": BEDROCK, "gateway": True},
            ValueError,
            "is a Bedrock inference profile; it cannot use the HUD gateway",
            id="claude-bedrock-through-the-gateway",
        ),
        pytest.param(
            OPENAI, NO_KEY, {}, HudAuthenticationError, "No API key for openai", id="openai-no-key"
        ),
        pytest.param(
            GEMINI, NO_KEY, {}, HudAuthenticationError, "No API key for gemini", id="gemini-no-key"
        ),
        pytest.param(CHAT, NO_KEY, {}, ValueError, "No API key found", id="chat-no-key"),
        pytest.param(
            CHAT,
            {},
            {"api_key": "other", "base_url": "<gateway>"},
            ValueError,
            "api_key is not allowed with HUD Gateway",
            id="chat-a-foreign-key-for-the-hud-gateway",
        ),
    ],
)
def test_an_agent_with_no_route_to_its_model_is_refused_when_built(
    family: Provider,
    variables: dict[str, str | None],
    config: dict[str, Any],
    error: type[Exception],
    message: str,
    models: Models,
    hud_env: HudEnv,
) -> None:
    hud_env.set(**variables)

    with pytest.raises(error, match=message):
        family.agent(**{k: models.url if v == "<gateway>" else v for k, v in config.items()})


@EVERY_PROVIDER
async def test_each_gateway_request_carries_its_own_runs_trace_id(
    family: Provider, models: Models, tmp_path: Path
) -> None:
    models.script([say("done")])
    env = workspace_env(tmp_path / "ws")

    job = await Taskset("t", [Task(env=env.name, id="task")]).run(
        family.agent(), runtime=LocalRuntime(env), group=3
    )

    sent = sorted(uuid.UUID(request.headers["trace-id"]).hex for request in models.requests())
    assert sent == sorted(uuid.UUID(str(run.trace_id)).hex for run in job.runs)
