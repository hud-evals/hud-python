"""ModalRuntime on a real Modal account, and what it refuses before it creates a sandbox.

The real scenarios build the fixture image from this checkout on the account the
developer or CI configured; the refusals never reach Modal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import modal
import pytest

from hud.eval import (
    ComposeProject,
    ModalRuntime,
    RuntimeConfig,
    RuntimeGPU,
    RuntimeLimits,
    RuntimeResources,
    Task,
    rollout,
)
from tests.harness import ScriptedAgent

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tests.harness import HudEnv

ECHO = Task(env="echo", id="echo", args={"word": "tangerine"})


@pytest.mark.e2e
@pytest.mark.modal
async def test_a_verifier_in_its_own_sandbox_receives_the_actors_session_files(
    modal_image: Callable[[str], Any], modal_account: HudEnv
) -> None:
    """The verifier's own limits give it its own sandbox, so the actor's files must move."""
    own = RuntimeConfig(limits=RuntimeLimits(startup_timeout_s=600))
    row = Task(
        env="sessions", id="solve", verifier=Task(env="sessions", id="verify", runtime_config=own)
    )

    run = await rollout(
        row,
        ScriptedAgent("secret"),
        runtime=ModalRuntime(image=modal_image("sessions"), workdir="/hud"),
    )

    assert run.reward == 1.0


def compose(tmp_path: Path) -> ComposeProject:
    path = tmp_path / "compose.yaml"
    path.write_text("services:\n  main: {image: lab}\n")
    return ComposeProject(document=path)


@pytest.mark.parametrize(
    ("options", "config", "message"),
    [
        pytest.param(
            {"image_name": "hud-lab"},
            RuntimeConfig(resources=RuntimeResources(gpu=RuntimeGPU(type=["A100", "H100"]))),
            "alternative GPU types",
            id="alternative-gpu-types",
        ),
        pytest.param(
            {"image_name": "hud-lab"},
            RuntimeConfig(resources=RuntimeResources(os="windows")),
            "resources.os",
            id="an-os-requirement",
        ),
        pytest.param({}, None, "requires image=", id="no-image"),
        pytest.param({}, "compose+gpu", "cannot attach GPUs", id="a-gpu-for-a-compose-project"),
        pytest.param(
            {"sandbox_secrets": ["secret:api"]},
            "compose",
            "sandbox secrets require an image runtime",
            id="sandbox-secrets-for-a-compose-project",
        ),
    ],
)
async def test_modal_refuses_what_it_cannot_provision_before_reaching_modal(
    options: dict[str, Any], config: Any, message: str, tmp_path: Path
) -> None:
    if isinstance(config, str):
        gpu = RuntimeResources(gpu=RuntimeGPU()) if config == "compose+gpu" else None
        config = RuntimeConfig(compose=compose(tmp_path), resources=gpu)

    with pytest.raises(ValueError, match=message):
        async with ModalRuntime(**options)(ECHO.model_copy(update={"runtime_config": config})):
            pytest.fail("the acquisition should fail")


def test_modal_refuses_an_app_and_an_app_name_together() -> None:
    with pytest.raises(ValueError, match="either app or app_name"):
        ModalRuntime(app=modal.App("mine"), app_name="other")
