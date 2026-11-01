"""The HUD API key a remote runtime hands its environment, so the environment can report spans."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from hud.eval import Task
from hud.eval.runtime import DaytonaRuntime, DockerRuntime, ModalRuntime

if TYPE_CHECKING:
    from hud.eval.runtime import Provider
    from tests.conftest import HudEnv


class SandboxCreated(Exception):
    def __init__(self, kwargs: dict[str, Any]) -> None:
        self.kwargs = kwargs


@pytest.fixture
def modal_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    async def lookup(*args: Any, **kwargs: Any) -> None:
        return None

    async def create(*args: Any, **kwargs: Any) -> None:
        raise SandboxCreated(kwargs)

    fake = SimpleNamespace(
        Image=SimpleNamespace(from_name=lambda name: name),
        App=SimpleNamespace(lookup=SimpleNamespace(aio=lookup)),
        Sandbox=SimpleNamespace(create=SimpleNamespace(aio=create)),
        Probe=SimpleNamespace(with_tcp=lambda port: port),
    )
    monkeypatch.setitem(sys.modules, "modal", fake)


async def sandbox_env(**runtime: Any) -> dict[str, str]:
    with pytest.raises(SandboxCreated) as created:
        async with ModalRuntime("image", **runtime)(Task(env="env", id="task")):
            pass
    return created.value.kwargs["env"]


@pytest.mark.usefixtures("modal_sandbox")
async def test_the_sandbox_gets_the_hud_api_key_to_report_telemetry(hud_env: HudEnv) -> None:
    hud_env.set(HUD_API_KEY="k", HUD_TELEMETRY_ENABLED="1")

    assert await sandbox_env(env_vars={"MODE": "x"}) == {"MODE": "x", "HUD_API_KEY": "k"}


@pytest.mark.parametrize(
    "runtime", [DockerRuntime("image"), DaytonaRuntime("snapshot"), ModalRuntime("image")]
)
async def test_telemetry_without_a_key_to_forward_fails_instead_of_dropping_spans(
    runtime: Provider, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_TELEMETRY_ENABLED="1")

    with pytest.raises(RuntimeError, match="HUD_API_KEY"):
        async with runtime(Task(env="env", id="task")):
            pass
