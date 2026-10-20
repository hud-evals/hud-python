"""What only ``ClaudeAgent`` does: the computer tool each model gets, which stream errors
it retries, and the shape of the images it sends back."""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING

import pytest
from PIL import Image

from hud.agents import ClaudeAgent
from hud.agents.types import ClaudeConfig
from hud.capabilities import Capability
from tests.fixtures.environments import workspace_task
from tests.fixtures.models import Turn, call, stream_error
from tests.fixtures.screen import fake_screen

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import HudEnv
    from tests.fixtures.models import Models

MODEL = "claude-sonnet-4-6"


@pytest.fixture(autouse=True)
def signed_in(hud_env: HudEnv) -> None:
    hud_env.set(HUD_API_KEY="k")


def png(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def upright_after_rotation() -> bytes:
    """A 40x20 JPEG whose EXIF orientation says to show it rotated, as 20x40."""
    exif = Image.Exif()
    exif[274] = 6
    buffer = io.BytesIO()
    Image.new("RGB", (40, 20), "red").save(buffer, format="JPEG", exif=exif)
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("model", "beta", "kind", "sized"),
    [
        ("claude-sonnet-4-6", "computer-use-2025-11-24", "computer_20251124", True),
        ("claude-sonnet-4-5", "computer-use-2025-01-24", "computer_20250124", True),
        ("qwen/qwen3.8-max", None, None, False),
    ],
    ids=["4-6", "4-5", "a-model-without-native-computer-use"],
)
async def test_each_model_gets_the_computer_tool_it_supports_for_the_screen(
    model: str, beta: str | None, kind: str | None, sized: bool, models: Models, tmp_path: Path
) -> None:
    models.script([Turn(text="done")])

    async with fake_screen(width=200, height=100) as screen:
        task = workspace_task(tmp_path / "ws", capabilities=(Capability.rfb(url=screen.url),))
        await task().run(ClaudeAgent(ClaudeConfig(model=model)))

    (request,) = models.requests()
    (computer,) = [tool for tool in request.body["tools"] if tool["name"] == "computer"]
    assert (request.headers.get("anthropic-beta"), computer.get("type")) == (beta, kind)
    assert (computer.get("display_width_px"), computer.get("display_height_px")) == (
        (200, 100) if sized else (None, None)
    )
    assert sized or "display is 200x100 pixels" in computer["description"]


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
    (run,) = (await task().run(ClaudeAgent(ClaudeConfig(model=MODEL)))).runs

    assert (run.trace.status, len(models.requests())) == (status, requests)


@pytest.mark.parametrize(
    ("data", "shape"),
    [
        (png(40, 30), (40, 30)),
        (png(3000, 2000), (3000, 2000)),
        (upright_after_rotation(), (20, 40)),
    ],
    ids=["a-small-png", "a-wide-png", "a-jpeg-turned-upright-by-its-exif"],
)
async def test_an_image_the_editor_views_is_sent_within_claudes_limits_in_its_shape(
    data: bytes, shape: tuple[int, int], models: Models, tmp_path: Path
) -> None:
    root = tmp_path / "ws"
    models.script(
        [
            call("str_replace_based_edit_tool", command="view", path=f"{root}/file"),
            Turn(text="done"),
        ]
    )

    task = workspace_task(root, files={"file": data})
    await task().run(ClaudeAgent(ClaudeConfig(model=MODEL)))

    ((block,),) = [r["content"] for r in models.requests()[-1].body["messages"][-1]["content"]]
    with Image.open(io.BytesIO(base64.b64decode(block["source"]["data"]))) as sent:
        width, height = sent.size
    assert max(width, height) <= 1568 and width * height <= 1_150_000
    assert abs(width / height - shape[0] / shape[1]) < 0.01
