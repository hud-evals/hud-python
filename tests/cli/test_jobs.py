"""``hud jobs``: list jobs, a job's traces, and cancel rollouts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from tests.support import Hud, Platform

JOB_ID = "03dd2a73-d3df-4d10-a54a-e3d87c2d530d"
COMPACT_JOB_ID = "03dd2a73d3df4d10a54ae3d87c2d530d"
TRACE_ID = "00000000-0000-4000-a000-000000000002"


@pytest.fixture
def job(platform: Platform) -> list[dict[str, Any]]:
    """A job with two rollouts; returns its traces."""
    traces: list[dict[str, Any]] = [
        {"id": TRACE_ID, "status": "running", "start_time": "2026-10-01T00:00:01Z"},
        {"id": "00000000-0000-4000-a000-000000000003", "status": "completed", "reward": 0.75},
    ]
    platform.add_job(traces, id=JOB_ID, name="nightly", status="running")
    return traces


@pytest.mark.parametrize(("argv", "limit"), [([], "20"), (["list", "-n", "7"], "7")])
def test_jobs_lists_the_job_ids_it_asked_for(
    hud: Hud, platform: Platform, argv: list[str], limit: str
) -> None:
    ids = [platform.add_job(name=f"job-{n}", status="completed") for n in range(3)]

    result = hud("jobs", *argv, "--quiet")

    assert (result.exit_code, result.stdout.split()) == (0, ids), result
    (listed,) = platform.services.requests("api", "GET", "/v2/jobs")
    assert listed.query["limit"] == [limit]


def test_a_compact_job_id_reads_the_jobs_traces(
    hud: Hud, platform: Platform, job: list[dict[str, Any]]
) -> None:
    result = hud("jobs", COMPACT_JOB_ID, "--json", "--limit", "2")

    assert (result.exit_code, result.json) == (0, job), result
    (read,) = platform.services.requests("api", "GET", "/v2/jobs/{id}/traces")
    assert (read.params["id"], read.query["limit"]) == (JOB_ID, ["2"])


def test_a_malformed_job_reference_is_a_usage_error(hud: Hud, platform: Platform) -> None:
    result = hud("jobs", "get", "not-a-job", "--json")

    assert (result.exit_code, result.json["error"]) == (2, "usage"), result


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
def test_cancel_asks_the_platform_to_stop_what_it_names(
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
