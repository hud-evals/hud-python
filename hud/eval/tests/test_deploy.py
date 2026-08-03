"""``hud.eval.deploy`` — the platform build exchange, without the CLI."""

from __future__ import annotations

from typing import Any

import pytest

from hud.eval.deploy import BuildOutcome, await_build, trigger_build
from hud.utils.exceptions import HudRequestError
from hud.utils.platform import PlatformClient


class _FakePlatform(PlatformClient):
    """Records the trigger payload and answers status from a scripted list."""

    payload: dict[str, Any] | None = None
    statuses: list[dict[str, Any]] = []  # noqa: RUF012 - overwritten per instance
    asked: int = 0

    async def apost(self, path: str, *, json: Any = None) -> dict[str, Any]:
        assert path == "/builds/trigger"
        object.__setattr__(self, "payload", json)
        return {"id": "build-1", "registry_id": "registry-1"}

    async def aget(self, path: str) -> dict[str, Any]:
        assert path == "/builds/build-1/status"
        scripted = self.statuses[min(self.asked, len(self.statuses) - 1)]
        object.__setattr__(self, "asked", self.asked + 1)
        if isinstance(scripted, Exception):
            raise scripted
        return scripted


def _platform(statuses: list[Any] | None = None) -> _FakePlatform:
    platform = _FakePlatform("https://api.example", "key")
    object.__setattr__(platform, "statuses", statuses or [])
    object.__setattr__(platform, "asked", 0)
    return platform


async def test_trigger_sends_the_runtime_the_caller_asked_for() -> None:
    platform = _platform()

    build_id, registry_id = await trigger_build(
        platform, build_id="build-1", name="test-env", runtime="modal"
    )

    assert (build_id, registry_id) == ("build-1", "registry-1")
    assert platform.payload is not None
    assert platform.payload["runtime_provider"] == "modal"
    assert platform.payload["name"] == "test-env"


async def test_trigger_sends_the_runtime_config() -> None:
    runtime_config = {"resources": {"gpu": {"type": "A10G", "count": 1}}}
    platform = _platform()

    await trigger_build(
        platform,
        build_id="build-1",
        name="test-env",
        runtime="modal",
        runtime_config=runtime_config,
    )

    assert platform.payload is not None
    assert platform.payload["runtime_config"] == runtime_config


async def test_trigger_omits_what_the_caller_did_not_set() -> None:
    """An absent option is absent from the payload — not sent as a null the
    platform would have to interpret."""
    platform = _platform()

    await trigger_build(platform, build_id="build-1", name="test-env")

    assert platform.payload is not None
    assert not {
        "registry_id",
        "runtime_provider",
        "runtime_config",
        "environment_variables",
        "build_args",
        "build_secrets",
    } & set(platform.payload)


async def test_await_returns_the_first_terminal_status() -> None:
    platform = _platform([{"status": "IN_PROGRESS"}, {"status": "SUCCEEDED", "version": "3"}])

    final = await await_build(platform, "build-1", poll_interval=0)

    assert final["status"] == "SUCCEEDED"
    assert final["version"] == "3"


async def test_await_survives_a_transient_status_failure() -> None:
    """A build that is running is not affected by this side failing to ask
    about it, so a failed status call must not end the wait."""
    platform = _platform(
        [HudRequestError("boom", status_code=502), {"status": "SUCCEEDED"}],
    )

    final = await await_build(platform, "build-1", poll_interval=0)

    assert final["status"] == "SUCCEEDED"
    assert platform.asked == 2


async def test_await_gives_up_at_the_deadline() -> None:
    platform = _platform([{"status": "IN_PROGRESS"}])

    final = await await_build(platform, "build-1", poll_interval=0, max_wait=0)

    assert final["status"] == "TIMED_OUT"


@pytest.mark.parametrize(
    ("status", "succeeded"),
    [("SUCCEEDED", True), ("FAILED", False), ("TIMED_OUT", False)],
)
def test_outcome_reports_success_only_for_a_succeeded_build(status: str, succeeded: bool) -> None:
    outcome = BuildOutcome(build_id="b", registry_id="r", status=status)

    assert outcome.succeeded is succeeded
