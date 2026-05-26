"""V5 workflow-level public API contracts.

Import surface tests catch missing names. These tests cover the next layer:
cheap, no-network workflow shapes that users rely on when writing envs,
tasks, evals, agents, and graders.
"""

from __future__ import annotations

import inspect
from importlib import import_module
from typing import Any, cast

from mcp.types import TextContent, TextResourceContents
from pydantic import BaseModel

import hud
from hud import Environment
from hud.agents import MCPAgent, OpenAIAgent, OpenAIChatAgent, create_agent
from hud.agents.gemini import GeminiAgent
from hud.eval.context import EvalContext
from hud.eval.task import Task
from hud.native import Grade, contains, contains_all, contains_any, exact_match, f1_score
from hud.server import MCPRouter, MCPServer
from hud.services import ChatService
from hud.tools import (
    AnthropicComputerTool,
    ApplyPatchTool,
    GeminiComputerTool,
    HudComputerTool,
    OpenAIComputerTool,
    ShellTool,
)
from hud.tools.agent import AgentTool
from hud.tools.base import BaseHub, BaseTool
from hud.tools.coding import EditTool
from hud.tools.executors.base import BaseExecutor
from hud.tools.executors.xdo import XDOExecutor
from hud.tools.filesystem import GlobTool, GrepTool, ListTool, ReadTool
from hud.tools.playwright import PlaywrightTool
from hud.tools.types import AgentAnswer, ContentResult, EvaluationResult, SubScore
from hud.types import AgentResponse, MCPToolCall, MCPToolResult, Trace, TraceStep


def _assert_signature_contains(
    callable_obj: object,
    expected: tuple[str, ...],
) -> None:
    parameters = inspect.signature(cast("Any", callable_obj)).parameters
    missing = [name for name in expected if name not in parameters]
    assert not missing, f"{callable_obj!r} missing parameters: {missing}"


class _ContractTool(BaseTool):
    async def __call__(self) -> list[TextContent]:
        return [TextContent(text="ok", type="text")]


async def test_environment_authoring_workflow_entrypoints_are_usable() -> None:
    env = Environment("Contract Env", instructions="Exercise the public API contract.")

    for method_name in (
        "add_tool",
        "tool",
        "scenario",
        "resource",
        "shutdown",
        "mount",
        "include_router",
        "connect_image",
        "connect_hub",
        "connect_url",
        "connect_server",
        "initialize",
        "run",
        "serve",
        "http_app",
    ):
        assert callable(getattr(env, method_name))

    def decorated_tool() -> str:
        return "decorated"

    def added_tool() -> str:
        return "added"

    assert env.tool(decorated_tool) is decorated_tool
    assert env.add_tool(added_tool) is None
    assert env.http_app() is not None

    tools = await env.list_tools()
    assert {tool.name for tool in tools} >= {"decorated_tool", "added_tool"}


async def test_environment_decorator_forms_used_by_env_templates() -> None:
    env = Environment("Template Contract")

    @env.tool()
    def default_named_tool() -> str:
        return "default"

    @env.tool(name="custom_name")
    def custom_named_tool() -> str:
        return "custom"

    @env.resource("telemetry://live")
    def telemetry() -> str:
        return "live"

    @env.shutdown
    async def cleanup() -> None:
        return None

    @env.initialize
    async def initialize() -> None:
        return None

    tools = await env.list_tools()
    resources = await env.list_resources()
    resource_contents = await env.read_resource("telemetry://live")

    assert {tool.name for tool in tools} >= {"default_named_tool", "custom_name"}
    assert [str(resource.uri) for resource in resources] == ["telemetry://live"]
    assert isinstance(resource_contents[0], TextResourceContents)
    assert resource_contents[0].text == "live"
    assert env._shutdown_fn is cleanup
    assert env._initializer_fn is initialize


def test_environment_connection_and_run_signatures_cover_template_usage() -> None:
    env = Environment("Connection Contract")

    _assert_signature_contains(
        env.connect_image,
        (
            "image",
            "alias",
            "docker_args",
            "env_vars",
            "prefix",
            "include",
            "exclude",
            "transform",
        ),
    )
    _assert_signature_contains(
        env.connect_hub,
        ("slug", "alias", "prefix", "include", "exclude", "transform"),
    )
    _assert_signature_contains(
        env.connect_url,
        ("url", "headers", "alias", "prefix", "include", "exclude", "transform"),
    )
    _assert_signature_contains(env.connect_server, ("server", "prefix"))
    _assert_signature_contains(
        env.connect_mcp,
        ("config", "alias", "prefix", "include", "exclude", "transform"),
    )
    _assert_signature_contains(env.connect_mcp_config, ("mcp_config", "kwargs"))
    _assert_signature_contains(env.run, ("transport", "show_banner", "transport_kwargs"))
    _assert_signature_contains(env.submit, ("scenario", "answer", "session_id"))
    _assert_signature_contains(env.run_scenario_setup, ("scenario_name", "args", "session_id"))
    _assert_signature_contains(env.run_scenario_evaluate, ("scenario_name", "session_id"))


def test_environment_mcp_config_connectors_register_without_connecting() -> None:
    env = Environment("MCP Config Contract")

    assert (
        env.connect_mcp(
            {
                "filesystem": {
                    "command": "python",
                    "args": ["-m", "http.server"],
                }
            },
            alias="fs",
            prefix="fs",
            include=["read_file"],
            exclude=["debug"],
        )
        is env
    )
    assert (
        env.connect_mcp_config(
            {
                "git": {"command": "python", "args": ["-m", "http.server"]},
                "browser": {"command": "python", "args": ["-m", "http.server"]},
            },
            prefix="tool",
        )
        is env
    )

    assert set(env._connections) == {"fs", "git", "browser"}


async def test_environment_tool_registration_accepts_instances_and_schema_kwargs() -> None:
    env = Environment("Tool Registration Contract")
    tool = _ContractTool(name="direct_tool")

    assert env.tool(tool) is tool

    @env.tool(output_schema=None)
    def schema_free_tool() -> str:
        return "ok"

    tools = await env.list_tools()

    assert {tool.name for tool in tools} >= {"direct_tool", "schema_free_tool"}


async def test_environment_local_tool_call_workflow_runs_without_network() -> None:
    env = Environment("Call Contract")

    @env.tool
    def add(x: int, y: int) -> int:
        return x + y

    async with env:
        result = await env.call_tool("add", x=2, y=3)

    assert isinstance(result, MCPToolResult)
    assert str(result) == "✓ 5"


def test_environment_scenario_decorator_creates_task_factory() -> None:
    env = Environment("Scenario Contract")

    async def checkout(user_id: str = "alice"):
        yield f"Checkout for {user_id}"
        yield 1.0

    scenario = env.scenario("checkout")(checkout)
    task = scenario.task(user_id="bob")

    assert callable(scenario)
    assert callable(scenario.task)
    assert isinstance(task, Task)
    assert task.env is env
    assert task.scenario == "checkout"
    assert task.args == {"user_id": "bob"}


def test_environment_callable_task_factory_and_chat_scenarios() -> None:
    env = Environment(name="Callable Contract")

    async def ask(messages: list[dict[str, str]] | None = None):
        yield messages or "Ask me anything"
        yield 1.0

    scenario = env.scenario(name="ask", chat=True, exclude_tools=["admin_*"])(ask)
    task = env("ask", user_id="alice")
    blank_env = Environment()
    blank_task = blank_env()

    assert scenario.task().scenario == "ask"
    assert task.env is env
    assert task.scenario == "ask"
    assert task.args == {"user_id": "alice"}
    assert blank_env.name == "environment"
    assert blank_task.scenario is None
    assert blank_task.args == {}


def test_scenario_metadata_and_structured_answer_contract() -> None:
    class ResearchAnswer(BaseModel):
        final_answer: str

    env = Environment("Structured Scenario Contract")

    async def research(messages: list[dict[str, str]] | None = None, query: str = "hud"):
        answer: AgentAnswer[ResearchAnswer] = yield messages or f"Research {query}"
        yield EvaluationResult(reward=1.0, content=answer.content.final_answer)

    scenario = env.scenario(
        name="research",
        chat=True,
        required_env_vars=["SEARCH_API_KEY"],
        exclude_tools=["admin_*"],
        exclude_sources=["debug"],
        allowed_tools=["admin_status"],
        returns=ResearchAnswer,
        enable_citations=True,
    )(research)

    task = scenario.task(query="public api")
    wrapped_answer = AgentAnswer(
        content=ResearchAnswer(final_answer="done"),
        raw="done",
    )

    assert task.scenario == "research"
    assert task.args == {"query": "public api"}
    assert env._scenario_chat_flags["research"] is True
    assert env._scenario_output_config["research"] == (ResearchAnswer, True)
    assert env._scenario_exclusions["research"] == (
        ["admin_*"],
        ["debug"],
        ["admin_status"],
    )
    assert wrapped_answer.content.final_answer == "done"


def test_task_definition_workflow_accepts_validation_and_slug() -> None:
    env = Environment("Task Contract")
    task = Task(
        env=env,
        scenario="checkout",
        args={"user_id": "alice"},
        agent_config={"system_prompt": "Be precise."},
        metadata={"suite": "public-api"},
        columns={"difficulty": "easy", "score": 1.0},
    )
    validation = MCPToolCall(id="call_1", name="submit", arguments={"answer": "done"})

    task.validation = [validation]
    task.slug = "checkout-alice"
    task.agent_config = {"system_prompt": "Be careful."}
    task.metadata["owner"] = "sdk"

    assert task.env is env
    assert task.scenario == "checkout"
    assert task.args == {"user_id": "alice"}
    assert task.validation == [validation]
    assert task.slug == "checkout-alice"
    assert validation.id == "call_1"
    assert task.agent_config == {"system_prompt": "Be careful."}
    assert task.metadata == {"suite": "public-api", "owner": "sdk"}
    assert task.columns == {"difficulty": "easy", "score": 1.0}


def test_task_accepts_env_config_dict_for_hub_tasks() -> None:
    task = Task(env={"name": "browser", "include": ["navigate"], "exclude": ["debug"]})

    assert isinstance(task.env, Environment)
    assert task.env.name == "browser"
    assert task.env._hub_config == {
        "name": "browser",
        "include": ["navigate"],
        "exclude": ["debug"],
    }


def test_task_identity_validation_copy_and_model_dump_contract() -> None:
    env = Environment("Task Identity Contract").connect_hub("browser")
    task = Task(
        id="platform-task-version",
        slug="current-slug",
        env=env,
        scenario="checkout",
        args={"user_id": "alice"},
        validation=[MCPToolCall(name="submit", arguments={"answer": "done"})],
    )

    task.id = "mutated-task-version"
    cloned = task.copy(update={"slug": "copy-slug"})
    pydantic_clone = task.model_copy(update={"slug": "model-copy-slug"})
    dumped = task.model_dump(mode="python")
    validated = Task.model_validate(dumped)

    assert task.validation is not None
    assert task.validation[0].id
    assert task.id == "mutated-task-version"
    assert cloned.id is None
    assert cloned.slug == "copy-slug"
    assert pydantic_clone.id == "mutated-task-version"
    assert pydantic_clone.slug == "model-copy-slug"
    assert validated.scenario == "checkout"
    assert validated.args == {"user_id": "alice"}


async def test_eval_entrypoint_keeps_async_context_manager_contract() -> None:
    _assert_signature_contains(
        hud.eval,
        (
            "source",
            "name",
            "variants",
            "group",
            "group_ids",
            "job_id",
            "group_id",
            "trace_id",
            "api_key",
            "max_concurrent",
            "taskset_id",
            "trace",
            "quiet",
        ),
    )

    context_manager = hud.eval(quiet=True, trace=False)

    assert hasattr(context_manager, "__aenter__")
    assert hasattr(context_manager, "__aexit__")

    async with hud.eval(quiet=True, trace=False) as ctx:
        ctx.reward = 0.25

    assert ctx.reward == 0.25


def test_dataset_runner_entrypoints_keep_v5_signatures() -> None:
    datasets = import_module("hud.datasets")

    _assert_signature_contains(
        datasets.run_dataset,
        (
            "tasks",
            "agent_type",
            "agent_params",
            "max_steps",
            "max_concurrent",
            "group_size",
            "quiet",
            "job_id",
            "taskset_id",
        ),
    )
    _assert_signature_contains(datasets.load_tasks, ("source", "raw"))
    _assert_signature_contains(datasets.save_tasks, ("name", "tasks"))
    _assert_signature_contains(
        datasets.run_single_task,
        (
            "task",
            "agent_type",
            "agent_params",
            "max_steps",
            "job_id",
            "task_id",
            "group_id",
            "trace_name",
            "metadata",
            "trace_id",
            "api_key",
            "trace",
            "quiet",
        ),
    )
    _assert_signature_contains(
        datasets.submit_rollouts,
        (
            "tasks",
            "job_id",
            "agent_type",
            "agent_params",
            "max_steps",
            "group_size",
            "batch_size",
            "metadata",
        ),
    )
    _assert_signature_contains(
        datasets.display_results,
        (
            "results",
            "tasks",
            "name",
            "elapsed",
            "show_details",
        ),
    )


def test_agent_selection_contract_keeps_factory_and_run_methods() -> None:
    _assert_signature_contains(create_agent, ("model", "kwargs"))

    for agent_cls in (
        MCPAgent,
        OpenAIAgent,
        OpenAIChatAgent,
        GeminiAgent,
    ):
        assert callable(getattr(agent_cls, "create"))
        assert callable(getattr(agent_cls, "run"))
        _assert_signature_contains(agent_cls.run, ("ctx", "max_steps"))


def test_agent_response_and_factory_kwargs_contract() -> None:
    response = AgentResponse(content="done", done=True)

    assert response.content == "done"
    assert response.done is True

    _assert_signature_contains(OpenAIChatAgent.create, ("kwargs",))


async def test_mcp_server_lower_level_authoring_contract() -> None:
    server = MCPServer("Server Contract")

    @server.tool
    def ping() -> str:
        return "pong"

    tools = await server.list_tools()

    assert {tool.name for tool in tools} == {"ping"}


async def test_mcp_server_lifecycle_and_mount_contract() -> None:
    server = MCPServer("Server Lifecycle Contract", instructions="Serve tools.")
    nested = MCPServer("Nested Lifecycle Contract")
    hub = BaseHub("mounted")
    tool = _ContractTool(name="contract_tool")
    response_tool = _ContractTool(name="response")

    @server.initialize
    async def initialize() -> None:
        return None

    @server.shutdown
    async def shutdown() -> None:
        return None

    @server.resource("resource://status")
    def status() -> str:
        return "ok"

    server.add_tool(tool)
    server.add_tool(response_tool)
    server.mount(hub)
    server.mount(nested, prefix="nested")

    tools = await server.list_tools()
    resources = await server.list_resources()

    assert server.name == "Server Lifecycle Contract"
    assert callable(server.run)
    assert {tool.name for tool in tools} >= {"contract_tool", "response"}
    assert "resource://status" in {str(resource.uri) for resource in resources}


def test_mcp_server_run_and_lifecycle_signatures_cover_controller_usage() -> None:
    server = MCPServer("Server Signature Contract")

    _assert_signature_contains(MCPServer, ("name", "instructions", "fastmcp_kwargs"))
    _assert_signature_contains(server.run, ("transport", "show_banner", "transport_kwargs"))
    _assert_signature_contains(server.initialize, ("fn",))
    _assert_signature_contains(server.shutdown, ("fn",))
    _assert_signature_contains(server.mount, ("server", "namespace", "as_proxy", "prefix"))


async def test_base_hub_named_tool_decorator_contract() -> None:
    hub = BaseHub("evaluate")

    @hub.tool("table_match")
    def table_match(expected: str, actual: str) -> EvaluationResult:
        return EvaluationResult(reward=1.0 if expected == actual else 0.0)

    tools = await hub.list_tools()
    result = table_match("a", "a")

    assert {tool.name for tool in tools} == {"evaluate"}
    assert "tool:int_table_match@" in hub._local_provider._components
    assert result.reward == 1.0


async def test_mcp_router_tool_resource_prompt_composition_contract() -> None:
    router = MCPRouter()

    @router.tool()
    def ping() -> str:
        return "pong"

    @router.resource("resource://configs")
    def configs() -> str:
        return "cfg"

    @router.prompt()
    def prompt() -> str:
        return "hello"

    server = MCPServer("Router Contract")
    server.include_router(router, prefix="nested")

    tools = await server.list_tools()
    resources = await server.list_resources()
    prompts = await server.list_prompts()

    assert {tool.name for tool in tools} == {"nested_ping"}
    assert {resource.name for resource in resources} == {"nested_configs"}
    assert {prompt.name for prompt in prompts} == {"nested_prompt"}


async def test_environment_connect_server_and_base_tool_registration_contract() -> None:
    env = Environment("Connect Server Contract")
    server = MCPServer("Nested Contract")
    tool = _ContractTool(name="contract_tool", title="Contract Tool")

    @server.tool
    def ping() -> str:
        return "pong"

    env.connect_server(server, prefix="nested")
    env.add_tool(tool)

    tools = await env.list_tools()

    assert {tool.name for tool in tools} >= {"nested_ping", "contract_tool"}


async def test_environment_provider_format_helpers_resolve_registered_tools() -> None:
    env = Environment("Provider Format Contract")
    tool = _ContractTool(name="contract_tool", title="Contract Tool")

    env.add_tool(tool)
    await env.list_tools()

    assert [t.name for t in env.as_tools()] == ["contract_tool"]
    openai_tool = cast("dict[str, Any]", env.as_openai_chat_tools(strict=True)[0])
    assert openai_tool["function"]["name"] == "contract_tool"


def test_agent_tool_constructor_uses_task_template_contract() -> None:
    env = Environment("Agent Tool Contract")

    async def investigate(issue_id: str, expected_cause: str | None = None):
        yield f"Investigate {issue_id}"
        yield 1.0

    env.scenario("investigate")(investigate)
    agent_tool = AgentTool(
        env("investigate"),
        model="claude-haiku-4-5",
        name="investigate_issue",
        description="Investigate an issue",
    )

    assert agent_tool.name == "investigate_issue"
    assert agent_tool.description == "Investigate an issue"
    assert agent_tool.mcp.name == "investigate_issue"


async def test_grade_workflow_combines_subscores() -> None:
    result = await Grade.gather(SubScore(name="correct", value=1.0, weight=1.0))

    assert result.reward == 1.0
    assert result.done is True
    assert result.subscores is not None
    assert result.subscores[0].name == "correct"
    assert Grade.from_subscores([SubScore(name="partial", value=0.5, weight=1.0)]).reward == 0.5


def test_native_grader_helpers_keep_basic_semantics() -> None:
    assert exact_match(" France ", "france") == 1.0
    assert contains("hello world", "world") == 1.0
    assert contains_any("hello world", ["mars", "world"]) == 1.0
    assert contains_all("hello world", ["hello", "world"]) == 1.0
    assert f1_score("hello hud", "hello sdk") == 0.5


def test_eval_context_user_facing_properties_and_tool_helpers() -> None:
    ctx = EvalContext(trace=False, quiet=True, variants={"model": "test"})

    ctx.prompt = "Do the task"
    ctx.error = None
    ctx.results.append(EvalContext(trace=False, quiet=True))

    assert ctx.prompt == "Do the task"
    assert ctx.success is True
    assert callable(ctx.call_tool)
    assert callable(ctx.as_openai_chat_tools)
    assert ctx.variants == {"model": "test"}
    assert len(ctx.results) == 1

    ctx.error = RuntimeError("failed")
    assert ctx.success is False


def test_chat_service_session_api_contract() -> None:
    env = Environment("Chat Service Contract")
    task = Task(env=env, scenario="ask")
    service = ChatService(task, model="claude-haiku-4-5", trace=False)

    _assert_signature_contains(service.send, ("message", "session_id"))
    _assert_signature_contains(service.clear, ("session_id",))
    _assert_signature_contains(service.agent_card, ("url",))

    card = service.agent_card(url="http://localhost:8000/a2a")
    service.clear(session_id="alice")

    assert card.url == "http://localhost:8000/a2a"


async def test_base_tool_callbacks_and_base_hub_contract() -> None:
    hub = BaseHub("evaluate")
    tool = _ContractTool(name="callback_tool")
    calls: list[str] = []

    @tool.after
    async def record_after(result: object = None, **_: object) -> None:
        calls.append(str(result))

    tool.register(hub)
    result = await tool.mcp.run({})

    assert hub.name == "evaluate"
    assert result
    assert calls


def test_content_and_evaluation_result_contracts() -> None:
    combined = ContentResult(output="hello ", error="warn") + ContentResult(
        output="world",
        url="https://example.com",
    )
    image = ContentResult(base64_image="iVBORw0KGgo=")
    blocks = combined.to_content_blocks()
    evaluation = EvaluationResult(
        reward=0.5,
        done=False,
        content="partial",
        info={"reason": "partial"},
        isError=True,
        subscores=[SubScore(name="quality", value=0.5, weight=1.0)],
    )
    from_float = EvaluationResult.from_float(0.25)

    assert combined.output == "hello world"
    assert combined.error == "warn"
    assert combined.url == "https://example.com"
    assert [block.type for block in blocks] == ["text", "text", "text"]
    assert image.to_content_blocks()[0].type == "image"
    assert evaluation.reward == 0.5
    assert evaluation.done is False
    assert evaluation.info == {"reason": "partial"}
    assert evaluation.isError is True
    assert evaluation.subscores is not None
    assert evaluation.subscores[0].name == "quality"
    assert from_float.reward == 0.25
    assert from_float.done is True


def test_trace_model_dump_and_validate_contract() -> None:
    step = TraceStep(type="CLIENT", category="mcp", request={"name": "tool"})
    trace = Trace(content="done", trace=[step], messages=[{"role": "assistant"}])
    dumped = trace.model_dump()
    validated = Trace.model_validate(dumped)

    assert len(trace) == 1
    assert trace.num_messages == 1
    assert dumped["trace"][0]["request"] == {"name": "tool"}
    assert validated.trace[0].type == "CLIENT"


def test_tool_constructor_contracts_from_external_consumers() -> None:
    shell = ShellTool(cwd=".")
    patch = ApplyPatchTool(base_path=".")
    edit = EditTool()
    read = ReadTool(base_path=".")
    grep = GrepTool(base_path=".", max_results=10)
    glob = GlobTool(base_path=".", max_results=10)
    listing = ListTool(base_path=".", max_entries=10)

    assert shell.name == "bash"
    assert patch.name == "edit"
    assert edit.name == "edit"
    assert read.name == "read"
    assert grep.name == "grep"
    assert glob.name == "glob"
    assert listing.name == "list"


def test_computer_and_browser_tool_constructor_contracts() -> None:
    executor = BaseExecutor(display_num=99)
    hud_computer = HudComputerTool(executor=executor, width=800, height=600)
    openai_computer = OpenAIComputerTool(executor=executor, width=1024, height=768)
    anthropic_computer = AnthropicComputerTool(
        executor=executor,
        width=1400,
        height=850,
        screenshot_quality=75,
    )
    gemini_computer = GeminiComputerTool(executor=executor, width=1440, height=900)
    xdo = XDOExecutor(display_num=99)
    playwright = PlaywrightTool(cdp_url="http://localhost:9222")

    assert hud_computer.name == "computer"
    assert hud_computer.executor is executor
    assert openai_computer.width == 1024
    assert anthropic_computer.height == 850
    assert gemini_computer.width == 1440
    assert xdo.display_num == 99
    assert playwright.name == "playwright"


def test_telemetry_instrument_decorator_keeps_callable_shape() -> None:
    @hud.instrument(name="contract.sync")
    def sync_fn(value: int) -> int:
        return value + 1

    @hud.instrument(span_type="contract", record_args=False, record_result=False)
    def quiet_fn(value: int) -> int:
        return value

    @hud.instrument(span_type="agent", record_args=False, record_result=True)
    def agent_fn(value: int) -> int:
        return value

    assert sync_fn(1) == 2
    assert quiet_fn(1) == 1
    assert agent_fn(1) == 1
    assert getattr(sync_fn, "_hud_instrumented") is True


def test_global_settings_keep_public_url_and_key_attributes() -> None:
    settings_module = import_module("hud.settings")
    settings = settings_module.settings

    for attr in (
        "api_key",
        "hud_api_url",
        "hud_gateway_url",
        "hud_mcp_url",
        "hud_rl_url",
        "hud_telemetry_url",
        "hud_web_url",
    ):
        assert hasattr(settings, attr)
