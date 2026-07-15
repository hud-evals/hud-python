"""Reward statistics for completed evaluation jobs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .run import Run

_ZERO_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class GroupStats:
    """Reward spread and failures for one scheduler-assigned rollout group."""

    group_id: str
    run_count: int
    error_count: int
    reward_mean: float
    reward_std: float
    reward_min: float
    reward_max: float

    @classmethod
    def from_runs(cls, group_id: str, runs: Sequence[Run]) -> GroupStats:
        """Calculate statistics for a non-empty group of runs."""
        if not runs:
            raise ValueError("group statistics require at least one run")
        rewards = [run.reward for run in runs]
        return cls(
            group_id=group_id,
            run_count=len(runs),
            error_count=sum(run.trace.is_error for run in runs),
            reward_mean=mean(rewards),
            reward_std=pstdev(rewards) if len(rewards) > 1 else 0.0,
            reward_min=min(rewards),
            reward_max=max(rewards),
        )

    @property
    def is_informative(self) -> bool:
        """Whether repeated runs have non-zero reward spread."""
        return self.run_count > 1 and not math.isclose(
            self.reward_std, 0.0, abs_tol=_ZERO_TOLERANCE
        )

    @property
    def is_constant(self) -> bool:
        """Whether repeated runs all received the same reward."""
        return self.run_count > 1 and not self.is_informative

    @property
    def is_all_zero(self) -> bool:
        """Whether a repeated group received only zero rewards."""
        return self.is_constant and math.isclose(self.reward_max, 0.0, abs_tol=_ZERO_TOLERANCE)

    @property
    def is_all_one(self) -> bool:
        """Whether a repeated group received only rewards of one."""
        return self.is_constant and math.isclose(self.reward_min, 1.0, abs_tol=_ZERO_TOLERANCE)


@dataclass(frozen=True, slots=True)
class JobStats:
    """Aggregate reward and within-group statistics for a sequence of runs."""

    run_count: int
    error_count: int
    ungrouped_run_count: int
    reward_mean: float
    reward_std: float
    groups: tuple[GroupStats, ...]

    @classmethod
    def from_runs(cls, runs: Sequence[Run]) -> JobStats:
        """Calculate global and scheduler-grouped reward statistics."""
        rewards = [run.reward for run in runs]
        grouped: dict[str, list[Run]] = {}
        ungrouped_run_count = 0
        for run in runs:
            if run.group_id is None:
                ungrouped_run_count += 1
            else:
                grouped.setdefault(run.group_id, []).append(run)

        return cls(
            run_count=len(runs),
            error_count=sum(run.trace.is_error for run in runs),
            ungrouped_run_count=ungrouped_run_count,
            reward_mean=mean(rewards) if rewards else 0.0,
            reward_std=pstdev(rewards) if len(rewards) > 1 else 0.0,
            groups=tuple(
                GroupStats.from_runs(group_id, group) for group_id, group in grouped.items()
            ),
        )

    @property
    def group_count(self) -> int:
        """Number of scheduler-assigned groups represented by the runs."""
        return len(self.groups)

    @property
    def eligible_group_count(self) -> int:
        """Number of groups with at least two rewards to compare."""
        return sum(group.run_count > 1 for group in self.groups)

    @property
    def informative_group_count(self) -> int:
        """Number of repeated groups with non-zero reward spread."""
        return sum(group.is_informative for group in self.groups)

    @property
    def constant_group_count(self) -> int:
        """Number of repeated groups with zero reward spread."""
        return sum(group.is_constant for group in self.groups)

    @property
    def all_zero_group_count(self) -> int:
        """Number of repeated groups receiving only zero rewards."""
        return sum(group.is_all_zero for group in self.groups)

    @property
    def all_one_group_count(self) -> int:
        """Number of repeated groups receiving only rewards of one."""
        return sum(group.is_all_one for group in self.groups)

    @property
    def error_group_count(self) -> int:
        """Number of groups containing at least one errored run."""
        return sum(group.error_count > 0 for group in self.groups)

    @property
    def within_group_reward_std(self) -> float | None:
        """Mean reward standard deviation across groups with repeated runs."""
        eligible = [group.reward_std for group in self.groups if group.run_count > 1]
        return mean(eligible) if eligible else None

    @property
    def informative_group_rate(self) -> float | None:
        """Share of repeated groups with non-zero reward spread."""
        eligible = self.eligible_group_count
        return self.informative_group_count / eligible if eligible else None


__all__ = ["GroupStats", "JobStats"]
