"""The fakes speak the real protocols: each is driven here by the real SDK client or agent."""

from __future__ import annotations

import io
import json
import subprocess
from typing import TYPE_CHECKING

import httpx
from anthropic import AsyncAnthropic
from anthropic.types.beta import BetaMessageParam, BetaToolUseBlock
from dirty_equals import IsStr
from google import genai
from google.genai.types import HttpOptions
from inline_snapshot import snapshot
from openai import AsyncOpenAI
from openai.types.responses import ResponseFunctionToolCall
from PIL import Image

from hud import Environment
from hud.agents.openai_compatible import OpenAIChatAgent
from hud.agents.types import OpenAIChatConfig
from hud.capabilities import Capability, RFBClient
from hud.eval import LocalRuntime, Task, Taskset, rollout
from tests.harness import KeyEvent, Reply, ScriptedAgent, call, fake_screen, say, steps

if TYPE_CHECKING:
    from pathlib import Path

    from tests.harness import FakeDocker, FakeServices, Hud, HudEnv, Models

HEX_ID = IsStr(regex=r"[0-9a-f]{32}")


def test_undeclared_routes_answer_404_and_are_recorded(services: FakeServices) -> None:
    response = httpx.get(
        f"{services.url('api')}/v2/nothing?x=1", headers={"Authorization": "Bearer k"}
    )

    assert response.status_code == 404
    (request,) = services.requests("api", "GET", "/v2/nothing")
    assert request.query == {"x": ["1"]}
    assert request.bearer == "k"


def test_routes_serve_replies_in_order_then_repeat_the_last(services: FakeServices) -> None:
    services.route("api", "GET", "/v2/items/{id}", Reply(status=502), Reply(json={"ok": True}))
    url = f"{services.url('api')}/v2/items/7"

    assert [httpx.get(url).status_code for _ in range(3)] == [502, 200, 200]
    assert services.requests("api", path="/v2/items/{id}")[0].params == {"id": "7"}


def _sums() -> Environment:
    env = Environment("sums")

    @env.template()
    async def add(a: int, b: int):
        answer = yield f"add {a} {b}"
        yield 1.0 if answer == str(a + b) else 0.0

    return env


def _solve(prompt: str) -> str:
    return str(sum(int(x) for x in prompt.split()[1:]))


async def test_a_reporting_rollout_reaches_the_fake_platform(
    services: FakeServices, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_API_KEY="k", HUD_TELEMETRY_ENABLED="1")
    for path in ("/v2/trace/job/{id}/enter", "/v2/trace/{id}/enter", "/v2/trace/{id}/exit"):
        services.route("api", "POST", path, json={})
    services.route("telemetry", "POST", "/trace/{id}/telemetry-upload", json={})

    job = await Taskset("demo", [Task(env="sums", id="add", args={"a": 1, "b": 2})]).run(
        ScriptedAgent(_solve), runtime=LocalRuntime(_sums())
    )

    assert job.reward == 1.0
    assert services.bodies("api", "POST", "/v2/trace/{id}/exit") == snapshot(
        [{"status": "completed", "reward": 1.0, "evaluation_result": {"score": 1.0}}]
    )
    assert services.bodies("api", "POST", "/v2/trace/{id}/enter") == snapshot(
        [{"job_id": HEX_ID, "group_id": HEX_ID, "task_slug": "add-1744f53e", "model": "unknown"}]
    )
    (upload,) = services.requests("telemetry", "POST", "/trace/{id}/telemetry-upload")
    assert upload.bearer == "k"
    assert [span["name"] for span in upload.json["telemetry"]] == snapshot(
        ["step.task", "step.user", "step.task"]
    )


async def test_a_scripted_chat_model_drives_an_agent_through_a_workspace(
    models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    workspace = tmp_path / "workspace"
    report = workspace / "REPORT.md"
    env = Environment("report")
    env.workspace(workspace, guest_path=str(workspace))

    @env.initialize
    async def seed() -> None:
        workspace.mkdir(parents=True, exist_ok=True)

    @env.template()
    async def write_report():
        yield "Write PASS to REPORT.md."
        yield 1.0 if report.exists() and report.read_text().strip() == "PASS" else 0.0

    models.script([call("write", filePath=str(report), content="PASS"), say("done")])
    agent = OpenAIChatAgent(OpenAIChatConfig(model="scripted", max_steps=4))

    run = await rollout(Task(env="report", id="write_report"), agent, runtime=LocalRuntime(env))

    assert run.reward == 1.0
    first, second = models.requests()
    assert first.protocol == "chat"
    assert "write" in first.tools
    assert second.tool_results != []
    assert first.headers["authorization"] == "Bearer k"
    assert [step["source"] for step in steps(run.trace_id)] == snapshot(
        ["task", "user", "agent", "tool", "agent", "task"]
    )


async def test_a_scripted_anthropic_turn_streams_through_the_anthropic_sdk(models: Models) -> None:
    models.script([call("bash", command="ls"), say("done", reasoning="thinking it over")])
    client = AsyncAnthropic(base_url=models.url, api_key="k")
    messages: list[BetaMessageParam] = [{"role": "user", "content": "go"}]

    async with client.beta.messages.stream(model="m", max_tokens=64, messages=messages) as stream:
        first = await stream.get_final_message()
    (tool_use,) = first.content
    assert isinstance(tool_use, BetaToolUseBlock)
    messages += [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_use.id,
                    "name": tool_use.name,
                    "input": tool_use.input,
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_use.id, "content": "a.txt"}],
        },
    ]
    async with client.beta.messages.stream(model="m", max_tokens=64, messages=messages) as stream:
        second = await stream.get_final_message()

    assert first.stop_reason == "tool_use"
    assert (tool_use.name, tool_use.input) == ("bash", {"command": "ls"})
    assert [block.type for block in second.content] == ["thinking", "text"]
    assert models.requests()[1].tool_results == ["a.txt"]


async def test_scripted_responses_chain_turns_by_previous_response_id(models: Models) -> None:
    models.script([call("lookup", q="x"), say("found it")])
    client = AsyncOpenAI(base_url=models.url, api_key="k")

    first = await client.responses.create(model="m", input="find x")
    (function_call,) = first.output
    assert isinstance(function_call, ResponseFunctionToolCall)
    second = await client.responses.create(
        model="m",
        previous_response_id=first.id,
        input=[{"type": "function_call_output", "call_id": function_call.call_id, "output": "x=1"}],
    )

    assert function_call.name == "lookup"
    assert second.output_text == "found it"
    assert models.requests()[1].tool_results == ["x=1"]


async def test_a_scripted_gemini_turn_parses_in_the_genai_sdk(models: Models) -> None:
    models.script([call("click_at", x=10, y=20)])
    client = genai.Client(
        api_key="k", http_options=HttpOptions(base_url=models.url, api_version="v1beta")
    )

    response = await client.aio.models.generate_content(model="gemini-test", contents="click it")

    (function_call,) = response.function_calls or []
    assert (function_call.name, function_call.args) == ("click_at", {"x": 10, "y": 20})
    assert models.requests()[0].model == "gemini-test"


def test_the_hud_cli_runs_against_the_fake_platform(
    services: FakeServices, hud_env: HudEnv, hud: Hud
) -> None:
    hud_env.set(HUD_API_KEY="k")
    services.route("api", "GET", "/v2/jobs", json={"items": [{"id": "job-1", "name": "demo"}]})

    result = hud("jobs", "list", "--json")

    assert result.exit_code == 0, result
    assert result.json == [{"id": "job-1", "name": "demo"}]
    (request,) = services.requests("api", "GET", "/v2/jobs")
    assert request.query == {"limit": ["20"]}
    assert request.bearer == "k"


def test_fake_docker_logs_invocations_and_answers_from_rules(
    fake_docker: FakeDocker, tmp_path: Path
) -> None:
    compose = tmp_path / "compose.json"
    compose.write_text(json.dumps({"services": {}}))
    fake_docker.on(r"^port cid-1", stdout="127.0.0.1:43210\n")

    run = subprocess.run(["docker", "run", "image"], capture_output=True, text=True, check=True)
    port = subprocess.run(
        ["docker", "port", "cid-1", "8765"], capture_output=True, text=True, check=True
    )
    subprocess.run(["docker", "compose", "-f", str(compose), "up"], check=True)

    assert (run.stdout, port.stdout) == ("cid-1\n", "127.0.0.1:43210\n")
    assert fake_docker.commands() == [
        "run image",
        "port cid-1 8765",
        f"compose -f {compose} up",
    ]
    assert fake_docker.calls[-1].files == {str(compose): '{"services": {}}'}


async def test_the_fake_screen_serves_frames_and_records_input() -> None:
    async with fake_screen(width=8, height=6, color=(10, 20, 30)) as screen:
        client = await RFBClient.connect(Capability.rfb(url=screen.url))
        try:
            png, mime_type = await client.screenshot_png()
            client.conn.keyboard.press("a")
            client.conn.mouse.move(3, 4)
            await client.drain()
        finally:
            await client.close()

    assert mime_type == "image/png"
    image = Image.open(io.BytesIO(png)).convert("RGB")
    assert image.size == (8, 6)
    assert image.getpixel((0, 0)) == (10, 20, 30)
    assert KeyEvent(ord("a"), True) in screen.key_events()
    assert screen.pointer_events()[-1].x == 3
