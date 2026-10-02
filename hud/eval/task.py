"""Portable task rows and single-task execution."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SerializationInfo,
    field_serializer,
    field_validator,
)

from hud.environment.env import Environment
from hud.runtime import RuntimeConfig

if TYPE_CHECKING:
    from hud.agents.base import Agent
    from hud.runtime import HostedRuntime, Provider

    from .job import Job


class Task(BaseModel):
    """A task row: env name, task id and arguments; pure data, shared across rollouts."""

    model_config = ConfigDict(validate_assignment=True)

    _env: Environment | None = PrivateAttr(default=None)

    env: str = Field(min_length=1)
    id: str = Field(min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)
    slug: str = Field(
        default_factory=lambda data: (
            str(data.get("id", ""))
            + (
                "-"
                + hashlib.sha1(
                    json.dumps(args, sort_keys=True, default=str).encode("utf-8"),
                    usedforsecurity=False,
                ).hexdigest()[:8]
                if (args := data.get("args"))
                else ""
            )
        ),
        min_length=1,
    )
    validation: list[dict[str, Any]] | None = None
    agent_config: dict[str, Any] | None = None
    # Filter columns and leaderboard facets on the platform.
    columns: dict[str, Any] | None = None
    runtime_config: RuntimeConfig | None = None
    # An agent-less task graded after this one, with the same answer; its grade is the record.
    verifier: Task | None = None

    @field_validator("verifier")
    @classmethod
    def _reject_nested_verifier(cls, verifier: Task | None) -> Task | None:
        if verifier is not None and verifier.verifier is not None:
            raise ValueError("nested verifier tasks are not supported")
        return verifier

    @property
    def shares_verifier_runtime(self) -> bool:
        return (
            self.verifier is not None
            and self.verifier.env == self.env
            and self.verifier.runtime_config is None
        )

    @field_serializer("runtime_config")
    def _serialize_runtime_config(
        self,
        config: RuntimeConfig | None,
        info: SerializationInfo,
    ) -> dict[str, Any] | None:
        return (
            config.model_dump(
                mode=info.mode,
                exclude_unset=True,
                context=info.context,
            )
            if config is not None
            else None
        )

    async def run(
        self,
        agent: Agent,
        *,
        runtime: Provider | HostedRuntime | None = None,
        group: int | None = None,
        max_concurrent: int | None = None,
        job: Job | None = None,
        rollout_timeout: float | None = None,
    ) -> Job:
        """Run this task as a taskset of one (see ``Taskset.run``)."""
        from .taskset import Taskset  # circular: taskset -> sync -> task

        taskset = Taskset(self.slug, [self])
        return await taskset.run(
            agent,
            runtime=runtime,
            group=group,
            max_concurrent=max_concurrent,
            job=job,
            rollout_timeout=rollout_timeout,
        )
