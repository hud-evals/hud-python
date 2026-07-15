"""Group-aware reward diagnostics for evaluation jobs."""

from __future__ import annotations

import pytest

from hud.eval import Job, JobStats
from hud.eval.run import Run


def _run(reward: float, group_id: str | None, *, error: bool = False) -> Run:
    run = Run(None, "task", {})
    run.group_id = group_id
    run.grade.reward = reward
    run.trace.status = "error" if error else "completed"
    return run


def test_job_stats_exposes_between_group_variance_without_training_signal() -> None:
    job = Job(
        id="job",
        name="constant groups",
        group=2,
        runs=[
            _run(0.0, "zero"),
            _run(0.0, "zero"),
            _run(1.0, "one"),
            _run(1.0, "one"),
        ],
    )

    assert job.reward == pytest.approx(0.5)
    assert list(job.groups) == ["zero", "one"]
    assert job.stats.reward_std == pytest.approx(0.5)
    assert job.stats.within_group_reward_std == pytest.approx(0.0)
    assert job.stats.eligible_group_count == 2
    assert job.stats.informative_group_count == 0
    assert job.stats.constant_group_count == 2
    assert job.stats.all_zero_group_count == 1
    assert job.stats.all_one_group_count == 1
    assert job.stats.informative_group_rate == pytest.approx(0.0)


def test_job_stats_reports_mixed_groups_errors_and_ungrouped_runs() -> None:
    stats = JobStats.from_runs(
        [
            _run(0.0, "spread"),
            _run(1.0, "spread"),
            _run(0.25, "constant"),
            _run(0.25, "constant", error=True),
            _run(0.5, "singleton"),
            _run(0.75, None),
        ]
    )

    assert stats.run_count == 6
    assert stats.error_count == 1
    assert stats.ungrouped_run_count == 1
    assert stats.group_count == 3
    assert stats.eligible_group_count == 2
    assert stats.informative_group_count == 1
    assert stats.constant_group_count == 1
    assert stats.error_group_count == 1
    assert stats.within_group_reward_std == pytest.approx(0.25)
    assert stats.informative_group_rate == pytest.approx(0.5)


def test_job_stats_omit_group_rates_without_repeated_runs() -> None:
    empty = JobStats.from_runs([])
    singletons = JobStats.from_runs([_run(0.5, "group"), _run(0.25, None)])

    assert empty.reward_mean == 0.0
    assert empty.reward_std == 0.0
    assert empty.within_group_reward_std is None
    assert empty.informative_group_rate is None
    assert singletons.eligible_group_count == 0
    assert singletons.within_group_reward_std is None
    assert singletons.informative_group_rate is None
