"""``hud jobs``: list jobs, a job's traces, and cancel rollouts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from tests.support import Hud, Platform

JOB_ID = "03dd2a73-d3df-4d10-a54a-e3d87c2d530d"
COMPACT_JOB_ID = "03dd2a73d3df4d10a54ae3d87c2d530d"
TRACE_ID = "00000000-0000-4000-a000-000000000002"
UNTOUCHED = ["running", "queued", "completed"]


@pytest.fixture
def job(platform: Platform) -> list[dict[str, Any]]:
    """A job with one rollout running, one queued and one finished; returns its traces."""
    traces: list[dict[str, Any]] = [
        {"id": TRACE_ID, "status": "running", "start_time": "2026-10-01T00:00:01Z"},
        {"id": "00000000-0000-4000-a000-000000000003", "status": "queued"},
        {"id": "00000000-0000-4000-a000-000000000004", "status": "completed", "reward": 0.75},
    ]
    platform.add_job(traces, id=JOB_ID, name="nightly", status="running")
    return traces


@pytest.mark.parametrize(("argv", "shown"), [([], 20), (["list", "-n", "7"], 7)])
def test_jobs_lists_the_most_recent_job_ids(
    hud: Hud, platform: Platform, argv: list[str], shown: int
) -> None:
    ids = [platform.add_job(name=f"job-{n}", status="completed") for n in range(25)]

    result = hud("jobs", *argv, "--quiet")

    assert (result.exit_code, result.stdout.split()) == (0, ids[:shown]), result


def test_a_compact_job_id_reads_the_jobs_traces(hud: Hud, job: list[dict[str, Any]]) -> None:
    result = hud("jobs", COMPACT_JOB_ID, "--json", "--limit", "2")

    assert (result.exit_code, result.json) == (0, job[:2]), result


def test_a_malformed_job_reference_is_a_usage_error(hud: Hud, platform: Platform) -> None:
    result = hud("jobs", "get", "not-a-job", "--json")

    assert (result.exit_code, result.json["error"]) == (2, "usage"), result


# ─── cancel ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("argv", "statuses"),
    [
        pytest.param([JOB_ID, "--yes"], ["cancelled", "cancelled", "completed"], id="a-job"),
        pytest.param(
            [JOB_ID, "--trace-id", TRACE_ID], ["cancelled", "queued", "completed"], id="one-trace"
        ),
        pytest.param(["--all", "--yes"], ["cancelled", "cancelled", "completed"], id="every-job"),
    ],
)
def test_cancel_stops_the_live_rollouts_it_names(
    hud: Hud, job: list[dict[str, Any]], argv: list[str], statuses: list[str]
) -> None:
    result = hud("jobs", "cancel", *argv, "--json")

    assert result.exit_code == 0, result
    assert [trace["status"] for trace in job] == statuses


@pytest.mark.parametrize(
    "argv",
    [pytest.param([JOB_ID], id="a-job-unconfirmed"), pytest.param([], id="nothing-named")],
)
def test_cancel_refuses_without_a_confirmed_target(
    hud: Hud, job: list[dict[str, Any]], argv: list[str]
) -> None:
    result = hud("jobs", "cancel", *argv, "--json")

    assert (result.exit_code, result.json["error"]) == (2, "usage"), result
    assert [trace["status"] for trace in job] == UNTOUCHED


def test_a_cancel_dry_run_names_its_action_and_stops_nothing(
    hud: Hud, job: list[dict[str, Any]]
) -> None:
    result = hud("jobs", "cancel", JOB_ID, "--dry-run", "--json")

    assert (result.exit_code, result.json["action"]) == (0, "cancel_job"), result
    assert [trace["status"] for trace in job] == UNTOUCHED
