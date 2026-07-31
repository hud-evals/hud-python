"""Gold-patch validation against built instance images (Docker, linux/amd64).

The SWE-bench Pro analog of the official gold-patch sanity check, driven
through the full served lifecycle with **no agent edits**:

- ``validate_mode="golden"``: the dataset's golden patch grades to 1.0
- no validate_mode: an untouched baseline grades to 0.0

Requires images built by ``swe_tasks.py`` (skips otherwise)::

    uv run swe_tasks.py <instance_id>
    uv run pytest tests/test_integration.py -v
"""

import shutil

import pytest
from hud import DockerRuntime, Run, connect

from swe_tasks import tasks as ALL_TASKS

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio(loop_scope="session"),
]

ADAPTED = [t for t in ALL_TASKS if t.runtime_config and t.runtime_config.image]

if not shutil.which("docker"):
    pytest.skip("docker not available", allow_module_level=True)
if not ADAPTED:
    pytest.skip("no built instances (run swe_tasks.py first)", allow_module_level=True)


async def _grade(task, validate_mode: str | None) -> float:
    args = {"validate_mode": validate_mode} if validate_mode else {}
    runtime = DockerRuntime(task.runtime_config.image)
    async with runtime(task) as addr, connect(addr) as client:
        async with Run(client, task.id, args) as run:
            pass  # no agent work: setup on start, grading on exit
    return run.reward


@pytest.mark.parametrize("task", ADAPTED, ids=lambda t: t.slug)
async def test_golden_patch_resolves(task):
    assert await _grade(task, "golden") == 1.0


@pytest.mark.parametrize("task", ADAPTED, ids=lambda t: t.slug)
async def test_untouched_baseline_unresolved(task):
    assert await _grade(task, None) == 0.0
