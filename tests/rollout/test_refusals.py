"""What each runtime refuses to provision: a row it cannot place faithfully fails the run
before the agent sees a prompt, and nothing outside the process is asked for anything.

One table, a row per runtime and request it refuses. Docker runs with no
``docker`` on ``PATH``, Modal and Daytona with no reachable account, and the
HUD runtimes against the fake HUD services, which must see no request.
"""

from __future__ import annotations

import importlib.util
import re
import textwrap
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest

from hud.agents.openai_compatible import OpenAIChatAgent
from hud.agents.types import OpenAIChatConfig
from hud.eval import (
    ComposeProject,
    DaytonaRuntime,
    DockerRuntime,
    HostedRuntime,
    HUDRuntime,
    LocalRuntime,
    ModalRuntime,
    Runtime,
    RuntimeConfig,
    RuntimeGPU,
    RuntimeLimits,
    RuntimeResources,
    Shared,
    SubprocessRuntime,
    Task,
)
from tests.fixtures.envs import source
from tests.support import ScriptedAgent

from .placements import exported_runtimes

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from hud.agents.base import Agent
    from tests.support import FakeServices, HudEnv

NO_DOCKER = {"PATH": "/nonexistent"}
NO_DAYTONA = {"DAYTONA_API_KEY": "k", "DAYTONA_API_URL": "http://127.0.0.1:9"}


def compose(document: str, **extra: Any) -> Callable[[Path], RuntimeConfig]:
    def build(root: Path) -> RuntimeConfig:
        (root / ".env").write_text("")
        path = root / "compose.yaml"
        path.write_text(textwrap.dedent(document))
        return RuntimeConfig(compose=ComposeProject(document=path), **extra)

    return build


def config(**fields: Any) -> Callable[[Path], RuntimeConfig]:
    return lambda root: RuntimeConfig(**fields)


def gpu(**fields: Any) -> RuntimeResources:
    return RuntimeResources(gpu=RuntimeGPU(**fields))


def chat_agent() -> Agent:
    return OpenAIChatAgent(
        OpenAIChatConfig(model="test-model", api_key="secret", base_url="http://model.test")
    )


def needs(extra: str) -> pytest.MarkDecorator:
    return pytest.mark.skipif(
        importlib.util.find_spec(extra) is None, reason=f"needs the {extra} extra"
    )


def with_a_secret() -> ModalRuntime:
    import modal

    return ModalRuntime(sandbox_secrets=[modal.Secret.from_dict({"API": "x"})])


@dataclass(frozen=True)
class Refusal:
    runtime: Callable[[], Any]
    message: str
    config: Callable[[Path], RuntimeConfig | None] = lambda root: None
    variables: dict[str, str | None] = field(default_factory=dict)
    agent: Callable[[], Agent] = lambda: ScriptedAgent("x")


SIMPLE = "services:\n  main: {image: lab}\n"
CHAINED = """
services:
  main: {image: lab, network_mode: "service:gateway"}
  gateway: {image: gateway}
"""
REFUSALS: dict[type, dict[str, Refusal]] = {
    LocalRuntime: {
        "a runtime_config in process": Refusal(
            lambda: LocalRuntime(source("echo")),
            "LocalRuntime does not support task runtime_config",
            config(image="lab"),
        ),
    },
    SubprocessRuntime: {
        "a runtime_config in a child process": Refusal(
            lambda: SubprocessRuntime(source("echo")),
            "SubprocessRuntime does not support task runtime_config",
            config(image="lab"),
        ),
    },
    DockerRuntime: {
        "a GPU type": Refusal(
            DockerRuntime,
            "cannot select GPUs by type",
            config(image="lab:1", resources=gpu(type="H100")),
            NO_DOCKER,
        ),
        "run_args for a Compose project": Refusal(
            lambda: DockerRuntime(run_args=["--privileged"]),
            "run_args apply only to image environments",
            compose(CHAINED),
            NO_DOCKER,
        ),
        "a network cycle": Refusal(
            DockerRuntime,
            "network_mode service cycle includes 'main'",
            compose(
                "services:\n  main: {image: lab, network_mode: 'service:relay'}\n"
                "  relay: {image: relay, network_mode: 'service:main'}\n"
            ),
            NO_DOCKER,
        ),
        "a Compose include": Refusal(
            DockerRuntime,
            "does not support Compose include or extends",
            compose("include: [other.yaml]\n" + SIMPLE),
            NO_DOCKER,
        ),
        "a variable only the host sets": Refusal(
            DockerRuntime,
            "variable 'HOST_ONLY' is not set by the project .env",
            compose("services:\n  main: {image: $HOST_ONLY}\n"),
            NO_DOCKER,
        ),
    },
    ModalRuntime: {
        "alternative GPU types": Refusal(
            lambda: ModalRuntime("hud-lab"),
            "alternative GPU types",
            config(resources=gpu(type=["A100", "H100"])),
        ),
        "an OS requirement": Refusal(
            lambda: ModalRuntime("hud-lab"),
            "resources.os",
            config(resources=RuntimeResources(os="windows")),
        ),
        "a GPU for a Compose project": Refusal(
            ModalRuntime, "cannot attach GPUs", compose(SIMPLE, resources=gpu())
        ),
        "sandbox secrets for a Compose project": Refusal(
            with_a_secret,
            "sandbox secrets require an image runtime",
            compose(SIMPLE),
        ),
    },
    DaytonaRuntime: {
        "a Compose project": Refusal(
            lambda: DaytonaRuntime("snap"),
            "does not support runtime_config.compose",
            compose(SIMPLE),
            NO_DAYTONA,
        ),
        "a run limit": Refusal(
            lambda: DaytonaRuntime("snap"),
            "does not support runtime_config.run_timeout_s",
            config(limits=RuntimeLimits(run_timeout_s=60)),
            NO_DAYTONA,
        ),
        "an OS requirement": Refusal(
            DaytonaRuntime,
            "does not support runtime_config.resources.os",
            config(image="lab", resources=RuntimeResources(os="linux")),
            NO_DAYTONA,
        ),
        "a fraction of a CPU": Refusal(
            DaytonaRuntime,
            "needs a whole number of CPUs, got 1.5",
            config(image="lab", resources=RuntimeResources(cpu=1.5)),
            NO_DAYTONA,
        ),
        "neither a snapshot nor an image": Refusal(
            DaytonaRuntime, "requires snapshot_name or runtime_config.image", variables=NO_DAYTONA
        ),
        "resources for a snapshot already built": Refusal(
            lambda: DaytonaRuntime("snap"),
            "cannot resize an already-built snapshot",
            config(resources=RuntimeResources(cpu=2)),
            NO_DAYTONA,
        ),
    },
    HUDRuntime: {
        "a GPU the deployed env cannot have": Refusal(
            HUDRuntime,
            "HUDRuntime cannot honor this task's declared placement requirements or limits",
            config(resources=gpu(type="H100")),
        ),
        "no API key": Refusal(
            HUDRuntime, "HUD runtime tunnel requires HUD_API_KEY", variables={"HUD_API_KEY": None}
        ),
    },
    HostedRuntime: {
        "an agent the platform cannot run": Refusal(
            HostedRuntime, "hosted execution supports the registered agent types"
        ),
        "no API key": Refusal(
            HostedRuntime,
            "HUD_API_KEY is required",
            variables={"HUD_API_KEY": None},
            agent=chat_agent,
        ),
    },
}
NOT_APPLICABLE = {
    Runtime: "a borrowed runtime provisions nothing, so it has nothing to refuse",
    Shared: "refuses what its inner provider refuses",
}
EXTRAS: dict[type, str] = {ModalRuntime: "modal", DaytonaRuntime: "daytona"}


def test_every_exported_runtime_has_its_refusals_in_the_table() -> None:
    assert set(REFUSALS) | set(NOT_APPLICABLE) == exported_runtimes()


@pytest.mark.parametrize(
    "refusal",
    [
        pytest.param(
            refusal,
            id=f"{kind.__name__}-{name}",
            marks=[needs(EXTRAS[kind])] if kind in EXTRAS else [],
        )
        for kind, refusals in REFUSALS.items()
        for name, refusal in refusals.items()
    ],
)
async def test_a_row_a_runtime_cannot_provision_fails_before_the_prompt_and_asks_nothing(
    refusal: Refusal, services: FakeServices, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(**({"HUD_API_KEY": "k"} | refusal.variables))
    row = Task(env="echo", id="echo", args={"word": "x"}, runtime_config=refusal.config(tmp_path))

    job = await row.run(refusal.agent(), runtime=refusal.runtime())

    (run,) = job.runs
    assert (run.trace.status, run.prompt, job.errors) == ("error", None, [run])
    assert re.search(re.escape(refusal.message), run.trace.error or "")
    assert services.requests() == []
