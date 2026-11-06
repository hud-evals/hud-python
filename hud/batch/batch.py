"""Batch: template x dataset x binding, run lazily through warm environments."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from hud.environment.env import _TaskFactory
from hud.eval.taskset import Taskset

from .binding import Binding
from .dataset import Dataset
from .materialize import Materialization

if TYPE_CHECKING:
    from hud.agents.base import Agent
    from hud.eval import HostedRuntime, Job, Provider, Task

    from .dataset import Row
    from .ledger import Ledger


class Execution(BaseModel):
    """How a batch's rows run.

    ``agentic`` drives the agent's tool loop per row; ``one_shot`` makes one
    model call per row with the template's ``returns`` schema as the required
    reply shape (:attr:`~hud.agents.types.ToolAgentConfig.one_shot`). Rows
    stream through ``instances`` warm environment instances, each serving
    ``sessions_per_instance`` rows at once and retired after
    ``rows_per_instance`` rows or ``instance_ttl_s`` seconds. ``trace_sampling``
    is the fraction of rows that keep a full platform trace (chosen
    deterministically per row id); every row keeps a :class:`RowResult`, and
    failed rows keep their local trace when ``error_traces`` is set.
    """

    model_config = ConfigDict(frozen=True)

    mode: Literal["agentic", "one_shot"] = "agentic"
    instances: int = Field(default=1, ge=1)
    sessions_per_instance: int = Field(default=4, ge=1)
    rows_per_instance: int | None = Field(default=2000, ge=1)
    instance_ttl_s: float | None = Field(default=3 * 3600.0, gt=0)
    trace_sampling: float = Field(default=0.01, ge=0.0, le=1.0)
    error_traces: bool = True
    retries: int = Field(default=0, ge=0)
    max_rows: int | None = Field(default=None, ge=1)
    #: Halt when this fraction of the last 100 finished rows (at least 20) errored.
    stop_on_error_rate: float | None = Field(default=None, gt=0.0, le=1.0)
    rollout_timeout: float | None = Field(default=None, gt=0)

    @property
    def width(self) -> int:
        return self.instances * self.sessions_per_instance


class Batch(BaseModel):
    """A task template bound to a dataset: the definition of arbitrarily many tasks.

    Build one from a template (``Batch(name=..., template=scan, dataset=...,
    binding=...)``) or from its stored fields. :attr:`id` hashes what rows the
    batch produces (env, template, dataset, binding, shard layout), so two
    batches minting the same tasks share results and jobs on the same batch
    are comparable.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    env: str = Field(min_length=1)
    task: str = Field(min_length=1)
    dataset: Dataset
    binding: Binding
    materialization: Materialization = Field(default_factory=Materialization)
    execution: Execution = Field(default_factory=Execution)
    #: JSON Schema of the template's ``returns`` type; the one-shot reply shape.
    returns_schema: dict[str, Any] | None = None

    _template: _TaskFactory[...] | None = PrivateAttr(default=None)

    def __init__(self, *, template: _TaskFactory[...] | None = None, **data: Any) -> None:
        if template is not None:
            data.setdefault("env", template.env.name)
            data.setdefault("task", template.id)
            data.setdefault("returns_schema", template.manifest_entry().get("returns"))
        super().__init__(**data)
        if template is not None:
            if (self.env, self.task) != (template.env.name, template.id):
                raise ValueError("env/task disagree with the template")
            self.binding.validate_template(template)
            self._template = template

    @model_validator(mode="after")
    def _columns_cover_the_binding(self) -> Batch:
        if self.dataset.columns is not None:
            missing = self.binding.required_columns() - set(self.dataset.columns)
            # Locator fields are stamped by the reader, never selected.
            missing = {name for name in missing if not name.startswith("_")}
            if missing:
                raise ValueError(
                    f"binding reads columns the dataset does not select: {sorted(missing)}"
                )
        return self

    @property
    def template(self) -> _TaskFactory[...] | None:
        """The live template this batch was built from, when built in-process."""
        return self._template

    @property
    def id(self) -> str:
        identity = {
            "env": self.env,
            "task": self.task,
            "dataset": {"uri": self.dataset.uri, "columns": self.dataset.columns},
            "binding": self.binding.model_dump(mode="json", exclude_none=True),
            "shards": self.materialization.shards,
        }
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        return f"{self.name}-{digest[:12]}"

    # ─── rows ────────────────────────────────────────────────────────────

    def bind(self, row: Row) -> Task:
        """The task row for one dataset row (placed on the live template's env when known)."""
        task = self.binding.bind(self.id, self.env, self.task, row)
        if self._template is not None:
            task._env = self._template.env
        return task

    async def sample(self, n: int, *, seed: int = 1337) -> Taskset:
        """A :class:`Taskset` of ``n`` eligible rows, uniformly sampled (one dataset pass).

        The bridge back to today's flows: sync it, run it, QA it, put it on a
        leaderboard. The sample is deterministic for a given ``seed``.
        """
        if n < 1:
            raise ValueError("sample size must be >= 1")
        chosen: list[Row] = []
        seen = 0
        rng = random.Random(seed)  # noqa: S311 - sampling rows, not secrets
        async for row in self.dataset.iter_rows(where=self.binding.expression()):
            seen += 1
            if len(chosen) < n:
                chosen.append(row)
            else:
                slot = rng.randrange(seen)
                if slot < n:
                    chosen[slot] = row
        chosen.sort(key=lambda row: (row["_file_index"], row["_row_group"], row["_row"]))
        return Taskset(f"{self.name} sample {len(chosen)}", [self.bind(row) for row in chosen])

    # ─── execution ───────────────────────────────────────────────────────

    async def run(
        self,
        agent: Agent,
        *,
        runtime: Provider | HostedRuntime | None = None,
        ledger: Ledger | None = None,
        instances: int | None = None,
        job: Job | None = None,
    ) -> Job:
        """Run every eligible row (see :class:`~hud.batch.scheduler.BatchScheduler`)."""
        from .scheduler import BatchScheduler

        return await BatchScheduler(
            self, agent, runtime=runtime, ledger=ledger, instances=instances, job=job
        ).run()

    # ─── sources ─────────────────────────────────────────────────────────

    @classmethod
    def from_file(cls, path: str | Path, *, name: str | None = None) -> Batch:
        """The :class:`Batch` a ``.py`` source (or directory) defines, by ``name`` if several."""
        from hud.utils.modules import iter_modules

        source = Path(path).resolve()
        found: dict[int, Batch] = {}
        for module in iter_modules(source):
            for attribute in dir(module):
                if attribute.startswith("_"):
                    continue
                value = getattr(module, attribute, None)
                if isinstance(value, Batch):
                    found.setdefault(id(value), value)
        batches = [batch for batch in found.values() if name is None or batch.name == name]
        if not batches:
            wanted = f" named {name!r}" if name else ""
            raise ValueError(f"no Batch{wanted} found in {source}")
        if len(batches) > 1:
            names = ", ".join(sorted(batch.name for batch in batches))
            raise ValueError(f"{source} defines several batches ({names}); pass name=")
        return batches[0]


__all__ = ["Batch", "Execution"]
