"""``hud jobs``: list jobs, a job's traces, and cancel rollouts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from tests.harness import FakeServices, Hud

JOB_ID = "03dd2a73-d3df-4d10-a54a-e3d87c2d530d"
COMPACT_JOB_ID = "03dd2a73d3df4d10a54ae3d87c2d530d"
TRACE_ID = "00000000-0000-4000-a000-000000000002"
JOBS = [
    {
        "id": JOB_ID,
        "name": "nightly",
        "taskset_name": "browser",
        "status": "completed",
        "created_at": "2026-10-01T00:00:00Z",
    },
    {"id": "00000000-0000-4000-a000-000000000009", "name": None, "status": None},
]
TRACES = [
    {
        "id": TRACE_ID,
        "status": "completed",
        "reward": 0.75,
        "start_time": "2026-10-01T00:00:01Z",
        "error": None,
    },
    {"id": "00000000-0000-4000-a000-000000000003", "status": "error", "error": "x" * 60},
]


@pytest.fixture
def jobs(platform: FakeServices) -> FakeServices:
    platform.route("api", "GET", "/v2/jobs", json={"items": JOBS})
    platform.route("api", "GET", "/v2/jobs/{id}/traces", json={"items": TRACES})
    return platform


@pytest.mark.parametrize(
    ("argv", "limit"),
    [(["jobs", "--quiet"], "20"), (["jobs", "list", "-n", "7", "--quiet"], "7")],
)
def test_jobs_lists_recent_job_ids(
    hud: Hud, jobs: FakeServices, argv: list[str], limit: str
) -> None:
    result = hud(*argv)

    assert (result.exit_code, result.stdout.split()) == (0, [job["id"] for job in JOBS]), result
    (request,) = jobs.requests("api", "GET", "/v2/jobs")
    assert request.query == {"limit": [limit]}


def test_jobs_list_json_is_the_platform_items(hud: Hud, jobs: FakeServices) -> None:
    result = hud("jobs", "list", "--json")

    assert (result.exit_code, result.json) == (0, JOBS), result


def test_a_compact_job_id_reads_the_jobs_traces(hud: Hud, jobs: FakeServices) -> None:
    result = hud("jobs", COMPACT_JOB_ID, "--json", "--limit", "7")

    assert (result.exit_code, result.json) == (0, TRACES), result
    (request,) = jobs.requests("api", "GET", "/v2/jobs/{id}/traces")
    assert (request.path, request.query) == (f"/v2/jobs/{JOB_ID}/traces", {"limit": ["7"]})


def test_a_malformed_job_reference_is_a_usage_error(hud: Hud, platform: FakeServices) -> None:
    result = hud("jobs", "get", "not-a-job", "--json")

    assert result.exit_code == 2, result
    assert result.json["error"] == "usage"
    assert platform.requests("api", path="/v2/jobs{rest:path}") == []


# ─── cancel ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("argv", "path", "body", "reply"),
    [
        pytest.param(
            ["jobs", "cancel", JOB_ID, "--yes"],
            "/v2/rollouts/cancel_job",
            {"job_id": JOB_ID},
            {"cancelled": 3},
            id="a-job",
        ),
        pytest.param(
            ["jobs", "cancel", JOB_ID, "--trace-id", TRACE_ID],
            "/v2/rollouts/cancel",
            {"job_id": JOB_ID, "trace_id": TRACE_ID},
            {"status": "accepted"},
            id="one-trace",
        ),
        pytest.param(
            ["jobs", "cancel", "--all", "--yes"],
            "/v2/rollouts/cancel_user_jobs",
            {},
            {"jobs_cancelled": 1, "total_tasks_cancelled": 2, "job_details": []},
            id="every-job",
        ),
    ],
)
def test_cancel_posts_the_rollout_cancellation(
    hud: Hud,
    platform: FakeServices,
    argv: list[str],
    path: str,
    body: dict[str, Any],
    reply: dict[str, Any],
) -> None:
    platform.route("api", "POST", path, json=reply)

    result = hud(*argv, "--json")

    assert result.exit_code == 0, result
    assert result.json.items() >= reply.items()
    assert [request.path for request in platform.requests("api", "POST")] == [path]
    assert platform.bodies("api", "POST", path) == [body]


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["jobs", "cancel", JOB_ID], id="a-job-unconfirmed"),
        pytest.param(["jobs", "cancel"], id="nothing-named"),
    ],
)
def test_cancel_refuses_without_a_confirmed_target(
    hud: Hud, platform: FakeServices, argv: list[str]
) -> None:
    result = hud(*argv, "--json")

    assert (result.exit_code, result.json["error"]) == (2, "usage"), result
    assert platform.requests("api", "POST") == []


def test_a_cancel_dry_run_names_its_action_and_posts_nothing(
    hud: Hud, platform: FakeServices
) -> None:
    result = hud("jobs", "cancel", JOB_ID, "--dry-run", "--json")

    assert (result.exit_code, result.json["dry_run"], result.json["action"]) == (
        0,
        True,
        "cancel_job",
    )
    assert platform.requests("api", "POST") == []
