"""``hud jobs``: a job's traces, and cancelling a job, one trace, or every active job."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from tests.conftest import Hud
    from tests.fixtures.services import FakeServices

JOB_ID = "03dd2a73-d3df-4d10-a54a-e3d87c2d530d"
TRACE_ID = "00000000-0000-4000-a000-000000000002"


def test_a_dashless_job_id_reads_the_dashed_job(signed_in: FakeServices, hud: Hud) -> None:
    signed_in.route("api", "GET", f"/v2/jobs/{JOB_ID}/traces", json={"items": []})

    result = hud("jobs", JOB_ID.replace("-", ""), "--json")

    assert result.exit_code == 0, result
    assert len(signed_in.requests("api", "GET", f"/v2/jobs/{JOB_ID}/traces")) == 1


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
    "argv",
    [["jobs", "cancel"], ["jobs", "cancel", JOB_ID, "--all"]],
    ids=["nothing-named", "a-job-and-all"],
)
def test_cancel_needs_exactly_one_target(
    argv: list[str], signed_in: FakeServices, hud: Hud
) -> None:
    result = hud(*argv, "--yes", "--json")

    assert (result.exit_code, result.json["error"]) == (2, "usage"), result
    assert signed_in.requests("api", "POST") == []
