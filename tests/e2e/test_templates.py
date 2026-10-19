"""Every ``hud init`` template, from ``hud init`` to a graded rollout, the way a user takes it.

Each case generates a project from a template, installs the SDK from this
checkout into it, lists its tasks, passes the template's own lint and tests,
and runs one capped rollout of its first task with ``hud eval``. Three lanes run
that lifecycle:

- a scripted model behind the fake gateway, on every PR, or a cheap real model
  through the HUD gateway in the ``live`` lane: the model is a row;
- ``hosted``: deployed to HUD and run on its runtime, with the trace read back
  from the platform. The pre-release workflow runs it one template per job.

A template that serves an isolated workspace (coding) needs the sandbox lane
locally; its environment is installed under /usr, where its image puts it.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests.conftest import REAL_ENVIRONMENT
from tests.support import Reply, fake_screen, steps

from .lifecycle import evaluated, scripted
from .projects import ROWS, TEMPLATES, assert_lifecycle, generate

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tests.support import FakeServices, Hud, HudEnv, Uv
    from tests.support.machine import Result

SANDBOXED = {"coding"}
TAKES_DATA_FILES = {"argument-hints"}
EVAL = ("eval", ROWS, "claude", "--max-steps", "10", "-y", "--json")
LIVE_MODEL = "claude-haiku-4-5"
HOSTED_MODEL = "claude-sonnet-4-6"
REAL_MODEL = ("--gateway", "--config", "max_tokens=2048")


def scripted_model(
    request: pytest.FixtureRequest, template: str
) -> tuple[str | None, tuple[str, ...]]:
    """A scripted model behind the fake gateway, and a data file the fake platform serves."""
    services: FakeServices = request.getfixturevalue("services")
    request.getfixturevalue("models").respond(scripted)
    request.getfixturevalue("hud_env").set(HUD_API_KEY="k")
    services.route("api", "GET", "/v2/data/ci-notes", json={"filename": "notes.txt"})
    services.route(
        "api",
        "GET",
        "/v2/data/ci-notes/download",
        json={"url": f"{services.url('api')}/files/notes"},
    )
    services.route("api", "GET", "/files/notes", Reply(body=b"2, 3 and 5 are prime."))
    return ("ci-notes" if template in TAKES_DATA_FILES else None), ()


def real_model(request: pytest.FixtureRequest, template: str) -> tuple[str | None, tuple[str, ...]]:
    """A cheap real model through the HUD gateway, and the shared CI attachment."""
    request.getfixturevalue("live")
    data_file = (
        REAL_ENVIRONMENT.get("HUD_CI_DATA_FILE_ID") if template in TAKES_DATA_FILES else None
    )
    return data_file, (*REAL_MODEL, "--model", LIVE_MODEL)


MODELS = {"scripted": scripted_model, "live": real_model}
MODEL_LANES = {"scripted": [], "live": [pytest.mark.e2e, pytest.mark.live]}


def lane() -> list[Any]:
    """One case per template and model; templates serving an isolated workspace join the
    sandbox lane, and the real model runs in the live lane."""
    return [
        pytest.param(
            name,
            model,
            id=f"{name}-{model}",
            marks=[
                *marks,
                *([pytest.mark.e2e, pytest.mark.sandbox] if name in SANDBOXED else []),
            ],
        )
        for model, marks in MODEL_LANES.items()
        for name in TEMPLATES
    ]


@pytest.fixture
def environment(request: pytest.FixtureRequest) -> Iterator[Path | None]:
    """Where the project's Python goes: under /usr for a sandboxed template, else ``.venv``."""
    if request.node.callspec.params["template"] not in SANDBOXED:
        yield None
        return
    location = Path("/usr/local/share/hud-e2e") / uuid.uuid4().hex
    yield location
    shutil.rmtree(location, ignore_errors=True)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(("template", "model"), lane())
async def test_a_template_runs_from_init_to_a_graded_rollout(
    template: str,
    model: str,
    environment: Path | None,
    hud_env: HudEnv,
    hud: Hud,
    uv: Uv,
    sdk_wheel: Path,
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    data_file, model_args = MODELS[model](request, template)

    project = generate(
        hud, template, tmp_path / f"ci-{template}", wheel=sdk_wheel, uv=uv, environment=environment
    )
    listed = project.hud("task", "list", "--json")
    assert listed.exit_code == 0, listed
    assert listed.json, "the template lists no tasks"
    if (project.root / "tests").is_dir():
        project.check()
    project.write_rows(data_file)
    async with fake_screen() as screen:
        hud_env.set(VNC_PORT=str(screen.port))
        run = evaluated(await asyncio.to_thread(project.hud, *EVAL, *model_args, timeout=900))

    assert_lifecycle(steps(run["trace_id"]), data_file)


@pytest.mark.e2e
@pytest.mark.live
@pytest.mark.hosted
@pytest.mark.timeout(2400)
@pytest.mark.parametrize("template", TEMPLATES)
def test_a_deployed_template_runs_on_hud_and_its_trace_reads_back(
    template: str, live: HudEnv, hud: Hud, uv: Uv, sdk_wheel: Path, tmp_path: Path
) -> None:
    data_file = None
    if template in TAKES_DATA_FILES:
        data_file = REAL_ENVIRONMENT.get("HUD_CI_DATA_FILE_ID")
        assert data_file, "HUD_CI_DATA_FILE_ID must name the shared CI attachment"
    traces = Path(REAL_ENVIRONMENT.get("HUD_E2E_TRACE_DIR") or tmp_path / "traces") / template
    # A stable directory name keeps every run deploying to the same environment.
    project = generate(hud, template, tmp_path / f"ci-{template}", wheel=sdk_wheel, uv=uv)
    project.bundle_sdk_into_image()
    deployed = project.hud("deploy", "--no-env", timeout=1800)
    assert deployed.exit_code == 0, deployed
    project.write_rows(data_file)
    live.set(HUD_TELEMETRY_ENABLED="1", HUD_TELEMETRY_LOCAL_DIR=str(traces))

    run = evaluated(
        project.hud(*EVAL, *REAL_MODEL, "--model", HOSTED_MODEL, "--runtime", "hud", timeout=900)
    )

    assert_lifecycle(steps(run["trace_id"], traces), data_file)
    # Without a local span directory, `hud trace get` reads the trace from the platform.
    live.set(HUD_TELEMETRY_LOCAL_DIR=None)
    events = read_back(project.hud, run["trace_id"])
    (traces / "remote-events.json").write_text(json.dumps(events, indent=2))


def read_back(hud: Hud, trace_id: str, attempts: int = 12, interval: float = 5.0) -> list[Any]:
    """The platform's events for ``trace_id``, polled until they include the agent's messages."""
    for _ in range(attempts - 1):
        result = hud("trace", "get", trace_id, "--json")
        if agent_spoke(result):
            return result.json
        time.sleep(interval)
    result = hud("trace", "get", trace_id, "--json")
    assert agent_spoke(result), f"no agent events on the platform after {attempts} reads:\n{result}"
    return result.json


def agent_spoke(result: Result) -> bool:
    return result.exit_code == 0 and any(event["kind"] == "agent_message" for event in result.json)
