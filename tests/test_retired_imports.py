"""Import paths v0.6.23 exported keep working until hud v0.7.0, with a warning naming the new one.

``RETIRED`` lists v0.6.23's exports that left their module, beside the module that
holds them now.

Annotations here are evaluated at definition, so a template defined inside a test
can use helpers imported inside it.
"""

import subprocess
import sys
import warnings
from importlib import import_module

import pytest

from hud import Environment, Task
from hud.clients import connect
from hud.runtime import LocalRuntime
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
    ("hud", "SyncPlan", "hud.cli.tasksets"),
    ("hud", "instrument", "hud.telemetry"),
    *[("hud.eval", name, "hud") for name in ["Grade", "Job", "Run", "Task", "Taskset", "Trace"]],
    ("hud.eval", "rollout", "hud.eval.run"),
    ("hud.eval", "SyncPlan", "hud.cli.tasksets"),
    *[("hud.eval", name, "hud.runtime") for name in RUNTIME_NAMES],
    *[("hud.eval.runtime", name, "hud.runtime") for name in [*RUNTIME_NAMES, "DockerBindMount"]],
    ("hud.environment", "Capability", "hud.capabilities"),
    ("hud.environment", "Peer", "hud.environment.egress"),
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

    async with (
        LocalRuntime(env)(Task(env=env.name, id="review")) as runtime,
        connect(runtime) as client,
    ):
        (template,) = await client.list_tasks()

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
