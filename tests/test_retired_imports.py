"""Import paths v0.6.23 exported keep working until hud v0.7.0, with a warning naming the new one.

``RETIRED`` lists v0.6.23's exports that left their module, beside the module that
holds them now.

Annotations here are evaluated at definition, so a template defined inside a test
can use helpers imported inside it.
"""

import subprocess
import sys
import textwrap
import warnings
from importlib import import_module
from pathlib import Path
from typing import Any, ClassVar, Self

import pytest

from hud import Environment, Run, Task
from hud.agents.base import Agent
from hud.capabilities import Capability, CapabilityClient, register_client
from hud.clients import connect
from hud.environment.client import HudClient
from hud.runtime import Runtime, RuntimeConfig, SubprocessRuntime
from hud.types import Step, TextContent
from hud.utils.exceptions import HudDeprecationWarning

RUNTIME_NAMES = [
    "ComposeProject",
    "DockerRuntime",
    "HUDRuntime",
    "HostedRuntime",
    "LocalRuntime",
    "ModalRuntime",
    "Provider",
    "Runtime",
    "RuntimeConfig",
    "RuntimeGPU",
    "RuntimeLimits",
    "RuntimeResources",
    "RuntimeTPU",
    "Shared",
    "SubprocessRuntime",
]
ROOT_RUNTIME_NAMES = sorted(set(RUNTIME_NAMES) - {"ModalRuntime", "Provider", "Shared"})

RETIRED = [
    *[("hud", name, "hud.runtime") for name in ROOT_RUNTIME_NAMES],
    ("hud", "Connection", "hud.capabilities"),
    ("hud", "connect", "hud.clients"),
    ("hud.clients", "register_client", "hud.capabilities"),
    ("hud.clients", "HudProtocolError", "hud.environment"),
    *[("hud.clients", name, "hud.environment.client") for name in ["HudClient", "Manifest"]],
    ("hud", "SyncPlan", "hud.cli.tasksets"),
    ("hud", "instrument", "hud.telemetry"),
    ("hud", "TrainingClient", "hud.train"),
    *[("hud.eval", name, "hud") for name in ["Grade", "Job", "Run", "Task", "Taskset", "Trace"]],
    ("hud.eval", "rollout", "hud.eval.run"),
    ("hud.eval", "SyncPlan", "hud.cli.tasksets"),
    *[("hud.eval", name, "hud.runtime") for name in RUNTIME_NAMES],
    *[("hud.eval.runtime", name, "hud.runtime") for name in [*RUNTIME_NAMES, "DockerBindMount"]],
    ("hud.environment", "Capability", "hud.capabilities"),
    ("hud.environment", "Peer", "hud.environment.egress"),
    ("hud.environment", "WorkspaceRoute", "hud.environment.egress"),
    ("hud.environment", "DEFAULT_SYSTEM_MOUNTS", "hud.environment.workspace"),
    ("hud.environment", "MountKind", "hud.environment.workspace"),
    ("hud.types", "EvaluationResult", "hud.graders"),
    ("hud.types", "SubScore", "hud.graders"),
    ("hud.eval.sync", "SyncPlan", "hud.cli.tasksets"),
    ("hud.eval.sync", "TaskUploadResult", "hud.cli.tasksets"),
    ("hud.eval.sync", "diff", "hud.cli.tasksets"),
    ("hud.eval.sync", "task_upload_payload", "hud.cli.tasksets"),
    ("hud.eval.sync", "TasksetRecord", "hud.cli.tasksets"),
    ("hud.eval.sync", "upload_taskset", "hud.cli.tasksets"),
]


def test_every_retired_import_warns_with_the_new_path_and_returns_the_same_object() -> None:
    broken: list[str] = []
    for old, name, new in RETIRED:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            try:
                value = getattr(import_module(old), name)
            except AttributeError:
                broken.append(f"{old}.{name}: not importable")
                continue
        messages = [
            str(warning.message)
            for warning in record
            if warning.category is HudDeprecationWarning and warning.filename == __file__
        ]
        if not any(
            f"`{name}` from `{old}`" in message
            and "hud v0.7.0" in message
            and f"`{new}`" in message
            for message in messages
        ):
            broken.append(f"{old}.{name}: no warning naming `{new}` at the import")
        if value is not getattr(import_module(new), name):
            broken.append(f"{old}.{name}: not the object at `{new}`")

    assert broken == []


def test_the_sdk_imports_no_retired_path_of_its_own() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "error::FutureWarning",
            "-c",
            "import hud, hud.agents, hud.cli.__main__, hud.environment, hud.eval.run, hud.runtime",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr


async def test_a_template_written_with_the_retired_argument_helpers_keeps_its_hints() -> None:
    arguments = import_module("hud.environment.arguments")
    with pytest.warns(HudDeprecationWarning, match="x-hud-hint"):
        prompt_arg = arguments.PromptArg
        data_files_arg = arguments.DataFilesArg
        data_file_arg = arguments.DataFileArg
        grading_arg = arguments.GradingArg
    with pytest.warns(HudDeprecationWarning, match="`file_id`"):
        data_file_ref = arguments.DataFileRef

    env = Environment("retired-hints")

    @env.template(id="review")
    async def review(
        prompt: prompt_arg,
        attachments: data_files_arg[data_file_ref],
        report: data_file_arg[data_file_ref],
        criteria: grading_arg[dict],
    ):
        yield prompt
        yield 1.0

    (template,) = [task.manifest_entry() for task in env.tasks.values()]

    schema = template["args"]
    hints = {name: spec.get("x-hud-hint") for name, spec in schema["properties"].items()}
    assert hints == {
        "prompt": "prompt",
        "attachments": "data-files",
        "report": "data-file",
        "criteria": "grading",
    }
    assert schema["properties"]["attachments"]["items"]["$ref"] == "#/$defs/DataFileRef"
    assert set(schema["$defs"]["DataFileRef"]["properties"]) == {"file_id", "path"}


async def test_a_v0_6_23_connect_still_reaches_a_served_environment() -> None:
    source = import_module("tests.fixtures.envs").source("echo")
    with pytest.warns(HudDeprecationWarning, match=r"runtime\.connect\(\)"):
        async with (
            SubprocessRuntime(source)(Task(env="echo", id="echo")) as runtime,
            connect(runtime) as client,
        ):
            assert isinstance(client, HudClient)
            (template,) = await client.list_tasks()

    assert template["id"] == "echo"


@register_client
class _V0623EchoClient(CapabilityClient):
    protocol: ClassVar[str] = "v0623-echo/1"

    def __init__(self, capability: Capability) -> None:
        self.capability = capability
        self.closed = False

    @classmethod
    async def connect(cls, cap: Capability) -> Self:
        return cls(cap)

    async def close(self) -> None:
        self.closed = True


V0623_ENV = """
from hud import Environment
from hud.capabilities import Capability

env = Environment("v0623")
env.add_capability(Capability(name="echo", protocol="v0623-echo/1", url="tcp://127.0.0.1:9"))


@env.template(id="greet")
async def greet():
    answer = yield {"prompt": "say hi", "bindings": {"echo": {"slot": "a"}}}
    yield 1.0 if answer == "hi" else 0.0
"""


async def test_a_v0_6_23_agent_and_runtime_still_drive_a_run(tmp_path: Path) -> None:
    source = tmp_path / "env.py"
    source.write_text(textwrap.dedent(V0623_ENV))
    task = Task(env="v0623", id="greet")
    config = RuntimeConfig(image="v0623:latest")
    seen: dict[str, Any] = {}

    class V0623Agent(Agent):
        async def __call__(self, run: Run) -> None:
            seen["text"] = run.prompt_text
            seen["messages"] = run.prompt_messages
            seen["task_id"] = run.task_id
            seen["bindings"] = run.bindings
            seen["runtime_config"] = run.runtime_config
            seen["manifest"] = run.client.manifest
            seen["binding"] = run.client.binding("echo")
            seen["opened"] = [await run.client.open("echo"), await run.client.open("v0623-echo")]
            run.record(Step(source="system", extra={"by": "v0623"}))
            run.trace.content = "hi"

    async with SubprocessRuntime(source)(task) as served:
        with pytest.warns(HudDeprecationWarning) as record:
            v0623_runtime = Runtime(served.url, {"ready_timeout": 30}, config)
            job = await task.run(V0623Agent(), runtime=v0623_runtime)

    warned = {
        str(warning.message).split("`")[1]
        for warning in record
        if warning.filename == __file__ and "hud v0.7.0" in str(warning.message)
    }
    assert warned == {
        "Runtime.params",
        "Run.prompt_text",
        "Run.prompt_messages",
        "Run.task_id",
        "Run.bindings",
        "Run.runtime_config",
        "Run.client",
        "Run.record",
    }
    (trace,) = job.traces
    assert (trace.reward, v0623_runtime.ready_timeout) == (1.0, 30.0)
    assert seen["text"] == "say hi"
    assert seen["messages"][0].content == TextContent(type="text", text="say hi")
    assert (seen["task_id"], seen["bindings"]) == ("greet", {"echo": {"slot": "a"}})
    assert seen["runtime_config"] == config
    (echo,) = seen["manifest"].bindings
    assert (echo.name, echo.url) == ("echo", "tcp://127.0.0.1:9")
    assert seen["binding"].params["slot"] == "a"
    first, again = seen["opened"]
    assert first is again
    assert (first.capability.name, first.closed) == ("echo", True)
    assert any(step.extra == {"by": "v0623"} for step in trace.steps)
