"""What the provider agents do against a scripted provider: faults, retries and
truncation, concurrent rollouts, and each provider's own request and tool handling."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from anthropic import AsyncAnthropicBedrock
from fastmcp import FastMCP
from pydantic import ValidationError

from hud.agents import ClaudeAgent, GeminiAgent, OpenAIAgent
from hud.agents.claude.tools.hosted import ClaudeToolSearchTool
from hud.agents.openai.tools.hosted import OpenAIToolSearchTool
from hud.agents.types import ClaudeConfig, ClaudeEffort, GeminiConfig, OpenAIConfig, ToolStep
from hud.capabilities import Capability
from hud.types import AgentType, TextContent
from tests.fixtures.asgi import mcp_server
from tests.fixtures.environments import workspace_task
from tests.fixtures.models import Turn, call, stream_error
from tests.fixtures.screen import fake_screen
from tests.fixtures.services import Reply

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from hud.agents import Agent
    from tests.conftest import HudEnv
    from tests.fixtures.models import Models

PROVIDERS = [
    ("claude", "claude-sonnet-4-6"),
    ("openai", "gpt-5.6"),
    ("gemini", "gemini-3.1-pro-preview"),
    ("openai_compatible", "qwen3.6-plus"),
]
EVERY_PROVIDER = pytest.mark.parametrize(
    ("agent_type", "model"), PROVIDERS, ids=[row[0] for row in PROVIDERS]
)


@pytest.fixture(autouse=True)
def signed_in(hud_env: HudEnv) -> None:
    hud_env.set(HUD_API_KEY="hud-key")


@EVERY_PROVIDER
async def test_one_agent_drives_concurrent_rollouts_each_with_its_own_conversation(
    agent_type: str, model: str, models: Models, tmp_path: Path
) -> None:
    models.script([Turn(text="done")])
    kind = AgentType(agent_type).cls
    task, agent = workspace_task(tmp_path / "ws"), kind(kind.config_cls(model=model))

    jobs = await asyncio.gather(task(prompt="first").run(agent), task(prompt="second").run(agent))

    assert [trace.status for job in jobs for trace in job.traces] == ["completed", "completed"]
    assert sorted(request.prompt for request in models.requests()) == ["first", "second"]


@EVERY_PROVIDER
async def test_a_reply_cut_at_the_token_cap_ends_the_run_as_truncated(
    agent_type: str, model: str, models: Models, tmp_path: Path
) -> None:
    models.script([Turn(text="part", truncated=True)])
    kind = AgentType(agent_type).cls

    task = workspace_task(tmp_path / "ws")
    (trace,) = (await task().run(kind(kind.config_cls(model=model)))).traces

    assert (trace.status, trace.stop_reason, trace.content) == (
        "completed",
        "length",
        "part",
    )


@EVERY_PROVIDER
async def test_a_provider_fault_ends_the_run_with_its_error(
    agent_type: str, model: str, models: Models, tmp_path: Path
) -> None:
    models.script([Turn(fail=Reply(status=400, json={"error": {"message": "model unavailable"}}))])
    kind = AgentType(agent_type).cls

    task = workspace_task(tmp_path / "ws")
    (trace,) = (await task().run(kind(kind.config_cls(model=model)))).traces

    assert trace.status == "error"
    assert trace.error is not None
    assert "model unavailable" in trace.error


@EVERY_PROVIDER
async def test_an_overloaded_provider_is_asked_again(
    agent_type: str, model: str, models: Models, tmp_path: Path
) -> None:
    busy = Reply(status=503, headers={"retry-after-ms": "0"}, json={"error": {"message": "busy"}})
    models.queue([Turn(fail=busy), Turn(text="done")])
    kind = AgentType(agent_type).cls

    task = workspace_task(tmp_path / "ws")
    (trace,) = (await task().run(kind(kind.config_cls(model=model)))).traces

    assert (trace.status, trace.content, len(models.requests())) == (
        "completed",
        "done",
        2,
    )


CLAUDE_MODEL = "claude-sonnet-4-6"
GEMINI_MODEL = "gemini-3.1-pro-preview"


@pytest.mark.parametrize(
    ("errors", "kind", "status", "requests"),
    [
        (1, "overloaded_error", "completed", 2),
        (3, "gateway_timeout", "error", 3),
        (1, "invalid_request_error", "error", 1),
    ],
    ids=["a-transient-error-is-retried", "three-transient-errors-fail", "a-request-error-fails"],
)
async def test_an_error_inside_the_stream_is_retried_only_when_transient(
    errors: int, kind: str, status: str, requests: int, models: Models, tmp_path: Path
) -> None:
    models.queue(
        [stream_error(kind, headers={"retry-after-ms": "0"})] * errors + [Turn(text="done")]
    )

    task = workspace_task(tmp_path / "ws")
    (trace,) = (await task().run(ClaudeAgent(ClaudeConfig(model=CLAUDE_MODEL)))).traces

    assert (trace.status, len(models.requests())) == (status, requests)


@pytest.mark.parametrize(
    ("effort", "sent"),
    [("high", {"output_config": {"effort": "high"}}), (None, {})],
    ids=["set", "unset"],
)
@pytest.mark.parametrize("bedrock", [False, True], ids=["anthropic", "bedrock"])
async def test_the_configured_effort_reaches_claude_only_when_set(
    effort: ClaudeEffort | None,
    sent: dict[str, object],
    bedrock: bool,
    models: Models,
    tmp_path: Path,
) -> None:
    models.script([Turn(text="done")])
    client = (
        AsyncAnthropicBedrock(api_key="bedrock-key", aws_region="us-east-1", base_url=models.url)
        if bedrock
        else None
    )
    config = ClaudeConfig(model=CLAUDE_MODEL, reasoning_effort=effort, model_client=client)

    task = workspace_task(tmp_path / "ws")
    (trace,) = (await task().run(ClaudeAgent(config))).traces

    (request,) = models.requests()
    assert trace.status == "completed"
    assert {key: request.body[key] for key in request.body.keys() & {"output_config"}} == sent


async def test_an_excluded_computer_action_reaches_the_environment_tool_of_that_name(
    models: Models, tmp_path: Path
) -> None:
    notes = FastMCP("notes")
    typed: list[str] = []
    notes.tool(name="type", description="Record text.")(lambda text: typed.append(text) or "noted")
    models.script([call("type", text="hello"), Turn(text="done")])
    config = GeminiConfig(model=GEMINI_MODEL, excluded_predefined_functions=["type"])

    async with fake_screen() as screen, mcp_server(notes, name="notes") as tools:
        capabilities = (Capability.rfb(url=screen.url), tools)
        task = workspace_task(tmp_path / "ws", capabilities=capabilities)
        (trace,) = (await task().run(GeminiAgent(config))).traces

    assert (trace.status, typed, screen.key_events()) == ("completed", ["hello"], [])


async def test_a_computer_toolset_member_lands_on_the_screen_and_answers_as_the_toolset(
    models: Models, tmp_path: Path
) -> None:
    click = {
        "type": "tool_use",
        "id": "toolu_click",
        "name": "left_click",
        "toolset_name": "computer",
        "input": {"coordinate": [3, 4]},
    }
    models.script([Turn(native=(click,), finish="tool_use"), Turn(text="done")])

    async with fake_screen() as screen:
        task = workspace_task(tmp_path / "ws", capabilities=(Capability.rfb(url=screen.url),))
        (trace,) = (await task().run(ClaudeAgent(ClaudeConfig(model="claude-opus-5")))).traces

    (result,) = (
        block
        for message in models.requests()[-1].body["messages"]
        for block in message["content"]
        if block.get("type") == "tool_result"
    )
    assert trace.status == "completed"
    assert (result["toolset_name"], result.get("is_error", False)) == ("computer", False)
    assert [(e.x, e.y, e.buttons) for e in screen.pointer_events()] == [
        (3, 4, 0),
        (3, 4, 1),
        (3, 4, 0),
    ]


async def test_only_the_three_newest_computer_turns_keep_their_screenshots(
    models: Models, tmp_path: Path
) -> None:
    models.script([call("click", x=156, y=208)] * 4 + [Turn(text="done")])

    async with fake_screen() as screen:
        task = workspace_task(tmp_path / "ws", capabilities=(Capability.rfb(url=screen.url),))
        await task().run(GeminiAgent(GeminiConfig(model=GEMINI_MODEL)))

    responses = [
        part["functionResponse"]
        for content in models.requests()[-1].body["contents"]
        for part in content["parts"]
        if "functionResponse" in part
    ]
    assert [bool(response.get("parts")) for response in responses] == [False, True, True, True]


@pytest.mark.parametrize(
    ("turn", "error"),
    [
        (Turn(fail=Reply(json={"candidates": []})), "Gemini returned no candidates"),
        (Turn(finish="MALFORMED_FUNCTION_CALL"), "Provider returned a malformed function call"),
    ],
    ids=["no-candidates", "a-malformed-function-call"],
)
async def test_a_reply_it_cannot_use_ends_the_run_with_its_error(
    turn: Turn, error: str, models: Models, tmp_path: Path
) -> None:
    models.queue([turn])

    task = workspace_task(tmp_path / "ws")
    (trace,) = (await task().run(GeminiAgent(GeminiConfig(model=GEMINI_MODEL)))).traces

    assert (trace.status, len(models.requests())) == ("error", 1)
    assert trace.error is not None
    assert error in trace.error


# Strict mode offers each position's type; the Chat Completions subset has no unions,
# so a mixed tuple's items stay untyped.
@pytest.mark.parametrize(
    ("agent_type", "model", "expected"),
    [
        (agent_type, model, {"string", "integer"} if agent_type == "openai" else {None})
        for agent_type, model in PROVIDERS
        if agent_type in {"openai", "openai_compatible"}
    ],
    ids=["openai", "openai_compatible"],
)
async def test_a_tuple_argument_is_offered_with_every_type_its_positions_take(
    agent_type: str, model: str, expected: set[str | None], models: Models, tmp_path: Path
) -> None:
    labels = FastMCP("labels")

    @labels.tool(description="Label a pair.")
    def label(pair: tuple[str, int]) -> str:
        return f"{pair[0]}={pair[1]}"

    models.script([Turn(text="done")])
    kind = AgentType(agent_type).cls

    async with mcp_server(labels, name="labels") as tools:
        task = workspace_task(tmp_path / "ws", capabilities=(tools,))
        await task().run(kind(kind.config_cls(model=model)))

    (request,) = models.requests()
    (tool,) = [t for t in request.body["tools"] if (t.get("function") or t).get("name") == "label"]
    items = (tool.get("function") or tool)["parameters"]["properties"]["pair"].get("items", {})
    offered = {option.get("type") for option in items.get("anyOf", [items])}
    assert offered == expected


@pytest.mark.parametrize(
    "agent",
    [
        lambda: ClaudeAgent(
            ClaudeConfig(model=CLAUDE_MODEL, hosted_tools=[ClaudeToolSearchTool(threshold=1)])
        ),
        lambda: OpenAIAgent(
            OpenAIConfig(model="gpt-5.6", hosted_tools=[OpenAIToolSearchTool(threshold=1)])
        ),
    ],
    ids=["claude", "openai"],
)
async def test_tools_past_the_search_threshold_are_deferred_to_tool_search(
    agent: Callable[[], Agent], models: Models, tmp_path: Path
) -> None:
    notes = FastMCP("notes")
    notes.tool(name="first", description="First.")(lambda: "1")
    notes.tool(name="second", description="Second.")(lambda: "2")
    models.script([Turn(text="done")])

    async with mcp_server(notes, name="notes") as tools:
        task = workspace_task(tmp_path / "ws", capabilities=(tools,))
        await task().run(agent())

    (request,) = models.requests()
    deferred = {
        tool["name"]: tool.get("defer_loading")
        for tool in request.body["tools"]
        if tool.get("name") in {"first", "second"}
    }
    assert deferred == {"first": True, "second": True}


def test_an_openai_setting_the_installed_sdk_does_not_declare_is_refused() -> None:
    with pytest.raises(ValidationError, match=r"format\.strictness"):
        OpenAIConfig(
            text={
                "format": {"type": "json_schema", "name": "answer", "schema": {}, "strictness": 1}
            }
        )


def _workspace(tmp_path: Path) -> Path:
    """Fixture files, searched through a symlink to their directory."""
    files = tmp_path / "files"
    (files / "sub").mkdir(parents=True)
    root = tmp_path / "ws"
    root.symlink_to(files)
    (root / "a.ts").write_text("// noExplicitAny\n")
    (root / "b.tsx").write_text("// biome-ignore\n")
    (root / "notes.md").write_text("noExplicitAny\n")
    (root / "package.json").write_text('{"lint": "noExplicitAny"}\n')
    (root / "sub" / "firebase.json").write_text('{"lint": "biome-ignore"}\n')
    return root


async def _tool_output(agent_type: str, model: str, root: Path) -> list[str]:
    kind = AgentType(agent_type).cls
    (trace,) = (await workspace_task(root)().run(kind(kind.config_cls(model=model)))).traces
    (step,) = (step for step in trace.steps if isinstance(step, ToolStep))
    assert step.result is not None
    return "".join(
        block.text for block in step.result.content if isinstance(block, TextContent)
    ).splitlines()


@pytest.mark.parametrize(
    ("agent_type", "model", "path_key"),
    [("openai_compatible", "qwen3.6-plus", "path"), ("gemini", GEMINI_MODEL, "dir_path")],
    ids=["openai_compatible", "gemini"],
)
@pytest.mark.parametrize(
    ("pattern", "found"),
    [
        ("*.json", ["package.json", "sub/firebase.json"]),
        ("**/firebase.json", ["sub/firebase.json"]),
        ("**/*.ts*", ["a.ts", "b.tsx"]),
        ("sub/*", ["sub/firebase.json"]),
        ("*.{ts,md}", ["a.ts", "notes.md"]),
    ],
    ids=["a-base-name", "at-any-depth", "at-no-depth", "under-a-directory", "braces"],
)
async def test_a_glob_matches_names_at_any_depth_and_paths_below_the_search_root(
    pattern: str,
    found: list[str],
    agent_type: str,
    model: str,
    path_key: str,
    models: Models,
    tmp_path: Path,
) -> None:
    root = _workspace(tmp_path)
    models.script([call("glob", **{"pattern": pattern, path_key: str(root)}), Turn(text="done")])

    lines = await _tool_output(agent_type, model, root)

    assert (
        sorted(line.removeprefix(f"{root}/") for line in lines if line.startswith(f"{root}/"))
        == found
    )


@pytest.mark.parametrize(
    ("agent_type", "model", "tool", "path_key", "include_key"),
    [
        ("openai_compatible", "qwen3.6-plus", "grep", "path", "include"),
        ("gemini", GEMINI_MODEL, "grep_search", "dir_path", "include_pattern"),
    ],
    ids=["openai_compatible", "gemini"],
)
@pytest.mark.parametrize(
    ("include", "found"),
    [
        ("", ["a.ts", "b.tsx", "notes.md", "package.json", "sub/firebase.json"]),
        ("*.{ts,tsx}", ["a.ts", "b.tsx"]),
        ("**/*.json", ["package.json", "sub/firebase.json"]),
        ("sub/*.json", ["sub/firebase.json"]),
    ],
    ids=["every-file", "braces", "at-any-depth", "under-a-directory"],
)
async def test_a_search_takes_an_extended_regex_and_filters_files_by_glob(
    include: str,
    found: list[str],
    agent_type: str,
    model: str,
    tool: str,
    path_key: str,
    include_key: str,
    models: Models,
    tmp_path: Path,
) -> None:
    root = _workspace(tmp_path)
    search = {"pattern": "noExplicitAny|biome-ignore", path_key: str(root), include_key: include}
    models.script([call(tool, **search), Turn(text="done")])

    lines = await _tool_output(agent_type, model, root)

    assert (
        sorted(
            line.split(":")[0].removeprefix(f"{root}/")
            for line in lines
            if line.startswith(f"{root}/")
        )
        == found
    )


@pytest.mark.parametrize(
    ("agent_type", "model", "tool", "path_key"),
    [
        ("openai_compatible", "qwen3.6-plus", "grep", "path"),
        ("gemini", GEMINI_MODEL, "grep_search", "dir_path"),
    ],
    ids=["openai_compatible", "gemini"],
)
@pytest.mark.parametrize("pattern", [";", "{}"])
async def test_a_search_for_find_exec_syntax_reaches_grep(
    pattern: str,
    agent_type: str,
    model: str,
    tool: str,
    path_key: str,
    models: Models,
    tmp_path: Path,
) -> None:
    root = _workspace(tmp_path)
    (root / "loop.c").write_text("for (;;) {}\n")
    models.script([call(tool, **{"pattern": pattern, path_key: str(root)}), Turn(text="done")])

    lines = await _tool_output(agent_type, model, root)

    assert [line for line in lines if line.startswith(f"{root}/")] == [
        f"{root}/loop.c:1:for (;;) {{}}"
    ]
