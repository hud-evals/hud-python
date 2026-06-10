"""``configure``: ambient placement/schedule resolution for the rollout engine.

Precedence everywhere: explicit call-site argument > ambient ``configure``
scope > defaults (provision-by-env-name placement, group=1, no cap).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, cast

import pytest

from hud.environment import Environment
from hud.eval import RunConfig, Taskset, configure, task
from hud.eval.config import active

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from hud.agents.base import Agent
    from hud.environment.runtime import Runtime


def _provider(marker: str) -> Any:
    """A Provider whose acquisition fails with a recognizable marker."""

    @asynccontextmanager
    async def acquire(_task: Any) -> AsyncIterator[Runtime]:
        raise RuntimeError(marker)
        yield  # pragma: no cover

    return acquire


def test_scopes_merge_per_field_and_restore_on_exit() -> None:
    outer_placement = _provider("outer")
    assert active() == RunConfig()

    with configure(on=outer_placement, group=8):
        assert active().on is outer_placement
        assert active().group == 8

        with configure(group=4, max_concurrent=2):
            assert active().on is outer_placement  # inherited
            assert active() == RunConfig(on=outer_placement, group=4, max_concurrent=2)

        assert active().group == 8
        assert active().max_concurrent is None

    assert active() == RunConfig()


def test_run_config_validates_schedule_bounds() -> None:
    with pytest.raises(ValueError, match="group"):
        RunConfig(group=0)
    with pytest.raises(ValueError, match="max_concurrent"):
        RunConfig(max_concurrent=0)


async def test_task_run_uses_ambient_placement_and_explicit_overrides_it() -> None:
    row = task(Environment("e"), "solve", n=1)

    agent = cast("Agent", object())  # never invoked: placement fails first

    with configure(on=_provider("ambient-placement")):
        run = await row.run(agent)  # provider fails -> isolated failed Run
        assert run.trace.isError
        assert "ambient-placement" in (run.trace.content or "")

        run = await row.run(agent, on=_provider("explicit-placement"))
        assert "explicit-placement" in (run.trace.content or "")


async def test_session_is_plumbing_and_never_reads_ambient_state() -> None:
    row = task(Environment("hosted-env"), "solve", n=1)

    # Even inside a configure scope, a bare session provisions by env name
    # (ambient resolution belongs to the engine, not the lifecycle plumbing).
    with (
        configure(on=_provider("ambient-placement")),
        pytest.raises(NotImplementedError, match="hosted-env"),
    ):
        async with row.session():
            pass

    with pytest.raises(NotImplementedError, match="hosted-env"):
        async with row.session():
            pass


async def test_taskset_run_resolves_schedule_from_ambient_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hud.eval.rollout import Run

    seen: list[tuple[str | None, Any]] = []

    async def fake_rollout(
        task: Any, agent: Any, *, on: Any = None, job_id: Any, group_id: Any
    ) -> Run:
        seen.append((group_id, on))
        return Run.failed("stub")

    monkeypatch.setattr("hud.eval.taskset.rollout", fake_rollout)
    ts = Taskset("demo", [task(Environment("e"), "solve", n=1)])
    placement = _provider("scoped-placement")

    with configure(group=3, on=placement):
        job = await ts.run(agent=cast("Agent", object()))

    assert job.group == 3
    assert len(seen) == 3
    assert len({group_id for group_id, _ in seen}) == 1  # one GRPO group
    assert all(on is placement for _, on in seen)  # resolved placement reaches the atom

    seen.clear()
    with configure(group=3):
        await ts.run(agent=cast("Agent", object()), group=1)  # explicit beats ambient
    assert len(seen) == 1
    assert seen[0][1] is None  # no placement anywhere -> atom default (provision)
