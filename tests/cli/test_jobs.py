"""``hud jobs``: recent jobs, a job's traces, and cancelling a job, one trace, or every
active job."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from tests.conftest import Hud
    from tests.fixtures.services import FakeServices

JOB_ID = "03dd2a73-d3df-4d10-a54a-e3d87c2d530d"
TRACE_ID = "00000000-0000-4000-a000-000000000002"


@pytest.mark.parametrize("argv", [["jobs"], ["jobs", "list"]], ids=["bare", "list"])
def test_jobs_lists_recent_jobs(argv: list[str], signed_in: FakeServices, hud: Hud) -> None:
    jobs = [{"id": f"job-{n}", "name": f"run {n}", "status": "completed"} for n in range(2)]
    signed_in.route("api", "GET", "/v2/jobs", json={"items": jobs})

    listed = hud(*argv, "--limit", "5", "--json")
    quiet = hud(*argv, "--quiet")

    assert (listed.exit_code, [job["id"] for job in listed.json]) == (0, ["job-0", "job-1"])
    assert signed_in.requests("api", "GET", "/v2/jobs")[0].query["limit"] == ["5"]
    assert quiet.stdout.split() == ["job-0", "job-1"]


@pytest.mark.parametrize(
    "argv",
    [["jobs", "get", JOB_ID], ["jobs", JOB_ID.replace("-", "")]],
    ids=["get", "a-bare-id-without-dashes"],
)
def test_a_job_shows_its_traces(argv: list[str], signed_in: FakeServices, hud: Hud) -> None:
    traces = [{"id": TRACE_ID, "status": "completed", "reward": 1.0}]
    signed_in.route("api", "GET", f"/v2/jobs/{JOB_ID}/traces", json={"items": traces})

    result = hud(*argv, "--json")

    assert (result.exit_code, result.json) == (0, traces), result


def test_an_unknown_job_is_not_found(signed_in: FakeServices, hud: Hud) -> None:
    signed_in.route("api", "GET", f"/v2/jobs/{JOB_ID}/traces", status=404, json={"detail": "No"})

    result = hud("jobs", "get", JOB_ID, "--json")

    assert (result.exit_code, result.json["error"]) == (1, "not_found"), result
    assert result.json["input"] == {"job_id": JOB_ID}


@pytest.mark.parametrize(
    ("argv", "path", "body"),
    [
        (["jobs", "cancel", JOB_ID], "/v2/rollouts/cancel_job", {"job_id": JOB_ID}),
        (
            ["jobs", "cancel", JOB_ID, "--trace-id", TRACE_ID],
            "/v2/rollouts/cancel",
            {"job_id": JOB_ID, "trace_id": TRACE_ID},
        ),
        (["jobs", "cancel", "--all"], "/v2/rollouts/cancel_user_jobs", {}),
        (["cancel", JOB_ID], "/v2/rollouts/cancel_job", {"job_id": JOB_ID}),
    ],
    ids=["a-job", "one-trace", "every-active-job", "the-deprecated-cancel"],
)
def test_cancel_asks_the_platform_to_stop_what_it_names(
    argv: list[str], path: str, body: dict[str, Any], signed_in: FakeServices, hud: Hud
) -> None:
    signed_in.route("api", "POST", path, json={"cancelled": 2, "status": "accepted"})

    result = hud(*argv, "--yes", "--json")

    assert result.exit_code == 0, result
    assert signed_in.bodies("api", "POST", path) == [body]


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["jobs", "cancel"], "Provide a job_id or use --all to cancel all active jobs."),
        (["jobs", "cancel", JOB_ID, "--all"], "Cannot specify both job_id and --all."),
    ],
    ids=["nothing-named", "a-job-and-all"],
)
def test_cancel_needs_exactly_one_target(
    argv: list[str], message: str, signed_in: FakeServices, hud: Hud
) -> None:
    result = hud(*argv, "--yes", "--json")

    assert (result.exit_code, result.json["message"]) == (2, message), result
    assert signed_in.requests("api", "POST") == []
