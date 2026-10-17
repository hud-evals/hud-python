"""``hud jobs``: list jobs, a job's traces, and cancel rollouts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from tests.support import Hud, Platform

JOB_ID = "03dd2a73-d3df-4d10-a54a-e3d87c2d530d"
COMPACT_JOB_ID = "03dd2a73d3df4d10a54ae3d87c2d530d"
TRACE_ID = "00000000-0000-4000-a000-000000000002"


def test_jobs_without_a_verb_lists_the_jobs(hud: Hud, platform: Platform) -> None:
    ids = [platform.add_job(name=f"job-{n}", status="completed") for n in range(3)]

    result = hud("jobs", "--quiet")

    assert (result.exit_code, result.stdout.split()) == (0, ids), result


def test_a_compact_job_id_reads_the_jobs_traces(hud: Hud, platform: Platform) -> None:
    platform.add_job([{"id": TRACE_ID, "status": "running"}], id=JOB_ID)

    result = hud("jobs", COMPACT_JOB_ID, "--json")

    assert result.exit_code == 0, result
    (read,) = platform.services.requests("api", "GET", "/v2/jobs/{id}/traces")
    assert read.params["id"] == JOB_ID


# ─── cancel ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("argv", "cancelled"),
    [
        pytest.param([JOB_ID, "--yes"], ("job", {"job_id": JOB_ID}), id="a-job"),
        pytest.param(
            [JOB_ID, "--trace-id", TRACE_ID],
            ("trace", {"job_id": JOB_ID, "trace_id": TRACE_ID}),
            id="one-trace",
        ),
        pytest.param(["--all", "--yes"], ("all", {}), id="every-job"),
    ],
)
def test_cancel_stops_only_what_it_names(
    hud: Hud, platform: Platform, argv: list[str], cancelled: tuple[str, dict[str, Any]]
) -> None:
    result = hud("jobs", "cancel", *argv, "--json")

    assert result.exit_code == 0, result
    assert platform.cancellations == [cancelled]


@pytest.mark.parametrize(
    "argv",
    [pytest.param([JOB_ID], id="a-job-unconfirmed"), pytest.param([], id="nothing-named")],
)
def test_cancel_refuses_without_a_confirmed_target(
    hud: Hud, platform: Platform, argv: list[str]
) -> None:
    result = hud("jobs", "cancel", *argv, "--json")

    assert (result.exit_code, result.json["error"]) == (2, "usage"), result
    assert platform.cancellations == []


def test_a_cancel_dry_run_names_its_action_and_stops_nothing(hud: Hud, platform: Platform) -> None:
    result = hud("jobs", "cancel", JOB_ID, "--dry-run", "--json")

    assert (result.exit_code, result.json["action"]) == (0, "cancel_job"), result
    assert platform.cancellations == []
