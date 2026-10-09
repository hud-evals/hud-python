"""``hud jobs``: list jobs, a job's traces, and cancel rollouts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from inline_snapshot import snapshot

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


def test_jobs_lists_recent_jobs_as_a_table(hud: Hud, jobs: FakeServices) -> None:
    result = hud("jobs")

    assert result.exit_code == 0, result
    assert result.lines == snapshot(
        [
            "Recent Jobs",
            "ID Name Taskset Status Created",
            "03dd2a73-d3df-4d10-a54a-e3d87c2d530d nightly browser completed 2026-10-01T00:00:00Z",
            "00000000-0000-4000-a000-000000000009 - - -",
            "View: https://hud.example/jobs",
            "Tip: hud jobs get <id> to see traces for a specific job",
        ]
    )
    assert [request.query for request in jobs.requests("api", "GET", "/v2/jobs")] == [
        {"limit": ["20"]}
    ]


@pytest.mark.parametrize(
    ("argv", "stdout"),
    [
        (["jobs", "list", "--quiet"], f"{JOB_ID}\n00000000-0000-4000-a000-000000000009\n"),
        (["jobs", "-n", "7", "--quiet"], f"{JOB_ID}\n00000000-0000-4000-a000-000000000009\n"),
    ],
)
def test_jobs_quiet_prints_one_id_per_line(
    hud: Hud, jobs: FakeServices, argv: list[str], stdout: str
) -> None:
    result = hud(*argv)

    assert (result.exit_code, result.stdout) == (0, stdout)


def test_jobs_list_json_is_the_platform_items(hud: Hud, jobs: FakeServices) -> None:
    result = hud("jobs", "list", "-n", "7", "--json")

    assert result.exit_code == 0, result
    assert result.json == JOBS
    (request,) = jobs.requests("api", "GET", "/v2/jobs")
    assert request.query == {"limit": ["7"]}


def test_no_jobs_says_so(hud: Hud, platform: FakeServices) -> None:
    platform.route("api", "GET", "/v2/jobs", json={"items": []})

    result = hud("jobs", "list")

    assert (result.exit_code, result.stdout) == (0, "No jobs found.\n")


def test_a_job_lists_its_traces_with_a_canonical_link(hud: Hud, jobs: FakeServices) -> None:
    result = hud("jobs", COMPACT_JOB_ID)

    assert result.exit_code == 0, result
    assert result.lines == snapshot(
        [
            "Job Traces 03dd2a73-d3df-4d10-a54a-e3d87c2d530d",
            "Trace ID Status Reward Started Error",
            "00000000-0000-4000-a000-000000000002 completed 0.750 2026-10-01T00…",
            "00000000-0000-4000-a000-000000000003 error - xxxxxxxxxxxxxx…",
            "View: https://hud.example/jobs/03dd2a73-d3df-4d10-a54a-e3d87c2d530d",
            "Tip: hud trace get <trace_id> to inspect a specific rollout",
        ]
    )
    (request,) = jobs.requests("api", "GET", "/v2/jobs/{id}/traces")
    assert request.path == f"/v2/jobs/{JOB_ID}/traces"


@pytest.mark.parametrize(
    "argv",
    [
        ["jobs", JOB_ID, "--json", "--limit", "7"],
        ["jobs", "--json", "--limit", "7", JOB_ID],
        ["jobs", "--json", "--limit", "5", JOB_ID, "--limit", "7"],
        ["jobs", "get", JOB_ID, "--json", "--limit", "7"],
        ["jobs", "get", "--json", "--limit", "7", JOB_ID],
        ["jobs", "get", "--json", "--limit", "5", JOB_ID, "--limit", "7"],
    ],
)
def test_job_options_apply_on_either_side_of_the_id(
    hud: Hud, jobs: FakeServices, argv: list[str]
) -> None:
    result = hud(*argv)

    assert result.exit_code == 0, result
    assert result.json == TRACES
    (request,) = jobs.requests("api", "GET", "/v2/jobs/{id}/traces")
    assert request.query == {"limit": ["7"]}


def test_a_job_without_traces_links_to_it(hud: Hud, platform: FakeServices) -> None:
    platform.route("api", "GET", "/v2/jobs/{id}/traces", json={"items": []})

    result = hud("jobs", "get", JOB_ID)

    assert result.exit_code == 0, result
    assert result.lines == snapshot(
        [
            "No traces found for this job.",
            "View: https://hud.example/jobs/03dd2a73-d3df-4d10-a54a-e3d87c2d530d",
        ]
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["jobs", "00000000-0000-0000-0000-invalid", "--json"],
        ["jobs", "unknown", "--json"],
        ["jobs", "get", "not-a-job", "--json"],
    ],
)
def test_a_malformed_job_reference_is_a_usage_error(
    hud: Hud, platform: FakeServices, argv: list[str]
) -> None:
    result = hud(*argv)

    assert result.exit_code == 2, result
    assert result.json["error"] == "usage"
    assert platform.requests("api", path="/v2/jobs{rest:path}") == []


# ─── cancel ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("argv", "path", "body", "reply", "document"),
    [
        (
            ["jobs", "cancel", JOB_ID, "--yes"],
            "/v2/rollouts/cancel_job",
            {"job_id": JOB_ID},
            {"cancelled": 3},
            snapshot(
                {
                    "action": "cancel_job",
                    "job_id": "03dd2a73-d3df-4d10-a54a-e3d87c2d530d",
                    "trace_id": None,
                    "cancelled": 3,
                }
            ),
        ),
        (
            ["jobs", "cancel", JOB_ID, "--trace-id", TRACE_ID],
            "/v2/rollouts/cancel",
            {"job_id": JOB_ID, "trace_id": TRACE_ID},
            {"status": "accepted"},
            snapshot(
                {
                    "action": "cancel_trace",
                    "job_id": "03dd2a73-d3df-4d10-a54a-e3d87c2d530d",
                    "trace_id": "00000000-0000-4000-a000-000000000002",
                    "status": "accepted",
                }
            ),
        ),
        (
            ["jobs", "cancel", "--all", "--yes"],
            "/v2/rollouts/cancel_user_jobs",
            {},
            {
                "jobs_cancelled": 1,
                "total_tasks_cancelled": 2,
                "job_details": [{"job_id": JOB_ID, "cancelled": 2}],
            },
            snapshot(
                {
                    "action": "cancel_all",
                    "job_id": None,
                    "trace_id": None,
                    "jobs_cancelled": 1,
                    "total_tasks_cancelled": 2,
                    "job_details": [
                        {"job_id": "03dd2a73-d3df-4d10-a54a-e3d87c2d530d", "cancelled": 2}
                    ],
                }
            ),
        ),
        (
            ["cancel", JOB_ID, "--yes"],
            "/v2/rollouts/cancel_job",
            {"job_id": JOB_ID},
            {"cancelled": 0},
            snapshot(
                {
                    "action": "cancel_job",
                    "job_id": "03dd2a73-d3df-4d10-a54a-e3d87c2d530d",
                    "trace_id": None,
                    "cancelled": 0,
                }
            ),
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
    document: dict[str, Any],
) -> None:
    platform.route("api", "POST", path, json=reply)

    result = hud(*argv, "--json")

    assert result.exit_code == 0, result
    assert result.json == document
    assert [request.path for request in platform.requests("api", "POST")] == [path]
    assert platform.bodies("api", "POST", path) == [body]


@pytest.mark.parametrize(
    ("argv", "exit_code", "document"),
    [
        (
            ["jobs", "cancel", JOB_ID],
            2,
            snapshot(
                {
                    "error": "usage",
                    "message": "Confirmation required in a non-interactive terminal.",
                    "suggestion": "Re-run with --yes to continue.",
                }
            ),
        ),
        (
            ["jobs", "cancel", "--all"],
            2,
            snapshot(
                {
                    "error": "usage",
                    "message": "Confirmation required in a non-interactive terminal.",
                    "suggestion": "Re-run with --yes to continue.",
                }
            ),
        ),
        (
            ["jobs", "cancel"],
            2,
            snapshot(
                {
                    "error": "usage",
                    "message": "Provide a job_id or use --all to cancel all active jobs.",
                    "suggestion": "hud jobs cancel <job-id>   or   hud jobs cancel --all --yes",
                }
            ),
        ),
        (
            ["cancel", JOB_ID, "--all"],
            2,
            snapshot(
                {
                    "error": "usage",
                    "message": "Cannot specify both job_id and --all.",
                    "input": {"job_id": "03dd2a73-d3df-4d10-a54a-e3d87c2d530d", "all": True},
                    "suggestion": "Pass either a job id or --all, not both.",
                }
            ),
        ),
        (
            ["jobs", "cancel", JOB_ID, "--dry-run"],
            0,
            snapshot(
                {
                    "dry_run": True,
                    "action": "cancel_job",
                    "job_id": "03dd2a73-d3df-4d10-a54a-e3d87c2d530d",
                    "trace_id": None,
                    "all": False,
                }
            ),
        ),
        (
            ["cancel", "--all", "--dry-run"],
            0,
            snapshot(
                {
                    "dry_run": True,
                    "action": "cancel_all",
                    "job_id": None,
                    "trace_id": None,
                    "all": True,
                }
            ),
        ),
    ],
)
def test_cancel_without_confirmation_or_as_a_dry_run_posts_nothing(
    hud: Hud, platform: FakeServices, argv: list[str], exit_code: int, document: dict[str, Any]
) -> None:
    result = hud(*argv, "--json")

    assert result.exit_code == exit_code, result
    assert result.json == document
    assert platform.requests("api", "POST") == []
