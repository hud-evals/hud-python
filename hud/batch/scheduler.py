"""BatchScheduler: stream rows through warm instances, record one result per row."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import time
import uuid
from collections import deque
from typing import TYPE_CHECKING, Any

from hud.agents.types import ToolAgentConfig
from hud.eval.job import Job, job_enter
from hud.eval.run import rollout
from hud.eval.runtime import HostedRuntime, InstancePool, LocalRuntime
from hud.telemetry import flush
from hud.utils.time import now_iso

from .ledger import Ledger, RowResult
from .materialize import RowStream, cursor_of

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from hud.agents.base import Agent
    from hud.eval import Provider, Run

    from .batch import Batch
    from .dataset import Row

logger = logging.getLogger("hud.batch")

_ERROR_WINDOW = 100
_ERROR_WINDOW_MIN = 20


class BatchHalted(RuntimeError):
    """The scheduler stopped taking rows (error rate, ``max_rows``); the job is checkpointed."""

    def __init__(self, job: Job, reason: str) -> None:
        super().__init__(reason)
        self.job = job
        self.reason = reason


def sampled(slug: str, fraction: float) -> bool:
    """Whether ``slug`` is in the deterministic ``fraction`` of rows that keep a trace."""
    if fraction >= 1.0:
        return True
    if fraction <= 0.0:
        return False
    digest = hashlib.sha1(slug.encode(), usedforsecurity=False).digest()[:4]
    return int.from_bytes(digest, "big") / 2**32 < fraction


def one_shot_agent(agent: Agent, returns_schema: dict[str, Any] | None) -> Agent:
    """``agent`` reconfigured for one call per row, answering in the template's shape."""
    if not isinstance(agent.config, ToolAgentConfig):
        raise TypeError(
            f"one_shot execution needs a provider tool agent, got {type(agent).__name__}"
        )
    return type(agent).load({**agent.dump(), "one_shot": True, "response_schema": returns_schema})


class BatchScheduler:
    """Run a :class:`~hud.batch.Batch`: rows in, :class:`RowResult`s out, one job.

    Rows come from a :class:`RowStream` (prefetched, sharded, resumable); each
    becomes a task and runs through an :class:`~hud.eval.InstancePool` over
    ``runtime`` (or remotely through ``HostedRuntime``). Results land in the
    ledger as they finish; reader checkpoints are persisted after each row so a
    restart with the same ledger resumes. ``job.runs`` stays empty on purpose:
    the per-row record is the ledger, not a list of every ``Run`` in memory.
    """

    def __init__(
        self,
        batch: Batch,
        agent: Agent,
        *,
        runtime: Provider | HostedRuntime | None = None,
        ledger: Ledger | None = None,
        instances: int | None = None,
        job: Job | None = None,
    ) -> None:
        self.batch = batch
        self.execution = batch.execution
        self.agent = (
            one_shot_agent(agent, batch.returns_schema)
            if self.execution.mode == "one_shot"
            else agent
        )
        self.ledger = ledger if ledger is not None else Ledger()
        self.instances = instances if instances is not None else self.execution.instances
        self.job = job
        if isinstance(runtime, HostedRuntime):
            self.hosted: HostedRuntime | None = runtime
            self.pool: InstancePool | None = None
        else:
            inner = runtime
            if inner is None:
                if batch.template is None:
                    raise ValueError(
                        "no placement: pass runtime= (the Batch was not built from a live "
                        "template, so its environment cannot be served in-process)"
                    )
                inner = LocalRuntime(batch.template.env)
            self.hosted = None
            self.pool = InstancePool(
                inner,
                instances=self.instances,
                width=self.execution.sessions_per_instance,
                rows_per_instance=self.execution.rows_per_instance,
                ttl_s=self.execution.instance_ttl_s,
            )
        self._recent: deque[bool] = deque(maxlen=_ERROR_WINDOW)
        self._halt: str | None = None
        self._taken = 0

    # ─── one row ─────────────────────────────────────────────────────────

    async def _run_once(self, task: Any, job_id: str, report: bool) -> Run:
        if self.hosted is not None:
            return await self.hosted.run(
                task, self.agent, job_id=job_id, rollout_timeout=self.execution.rollout_timeout
            )
        assert self.pool is not None
        return await rollout(
            task,
            self.agent,
            runtime=self.pool,
            job_id=job_id,
            rollout_timeout=self.execution.rollout_timeout,
            report=report,
        )

    async def _row(self, reader: int, row: Row, job: Job, first_attempt: int) -> RowResult:
        task = self.batch.bind(row)
        report = self.hosted is not None or sampled(task.slug, self.execution.trace_sampling)
        attempt = first_attempt
        while True:
            started = time.monotonic()
            run = await self._run_once(task, job.id, report)
            status = run.trace.status or "completed"
            error = run.trace.error if run.trace.is_error else None
            if run.grade.is_error:
                status = "error"
                error = error or run.grade.content or "ungraded"
            keep_trace = status == "error" and self.execution.error_traces
            result = RowResult(
                batch_id=self.batch.id,
                row_id=self.batch.binding.row_id_of(row),
                slug=task.slug,
                attempt=attempt,
                status=status,
                reward=None if run.grade.is_error else run.reward,
                evaluation=run.evaluation,
                error=error,
                trace_id=run.trace_id,
                reported=report,
                steps=len(run.trace.steps),
                duration_s=time.monotonic() - started,
                reader=reader,
                cursor=cursor_of(row),
                finished_at=now_iso(),
                trace=run.trace.model_dump(mode="json") if keep_trace else None,
            )
            if status != "error" or attempt > self.execution.retries:
                return result
            logger.info("retrying %s (attempt %d): %s", task.slug, attempt + 1, error)
            attempt += 1

    # ─── the job ─────────────────────────────────────────────────────────

    def _skip(self, attempts: Mapping[str, tuple[int, str]]) -> Callable[[Row], bool]:
        binding, batch_id = self.batch.binding, self.batch.id

        def skip(row: Row) -> bool:
            previous = attempts.get(binding.slug(batch_id, binding.row_id_of(row)))
            if previous is None:
                return False
            attempt, status = previous
            return status == "completed" or attempt > self.execution.retries

        return skip

    def _check_halt(self) -> None:
        rate = self.execution.stop_on_error_rate
        if rate is None or len(self._recent) < _ERROR_WINDOW_MIN or self._halt is not None:
            return
        failed = sum(self._recent)
        if failed / len(self._recent) >= rate:
            self._halt = (
                f"error rate {failed}/{len(self._recent)} reached stop_on_error_rate={rate:g}"
            )

    async def run(self) -> Job:
        batch, execution = self.batch, self.execution
        job = self.job
        if job is None:
            job = Job(id=uuid.uuid4().hex, name=f"{batch.name} (batch)", batch_id=batch.id)
            await job_enter(job.id, name=job.name, group=1)
        attempts = self.ledger.attempts(batch.id)
        resumed = self.ledger.checkpoints(batch.id)
        stream = RowStream(
            batch.dataset,
            batch.materialization,
            columns=batch.dataset.columns,
            where=batch.binding.expression(),
            resume=resumed,
            skip=self._skip(attempts),
        )
        in_flight = asyncio.Semaphore(execution.width)
        running: set[asyncio.Task[None]] = set()

        async def one(reader: int, row: Row) -> None:
            slug = batch.binding.slug(batch.id, batch.binding.row_id_of(row))
            previous = attempts.get(slug)
            first_attempt = previous[0] + 1 if previous is not None else 1
            try:
                result = await self._row(reader, row, job, first_attempt)
                self.ledger.record(result)
                if result.status == "completed":
                    job.rows_done += 1
                else:
                    job.rows_failed += 1
                self._recent.append(result.status != "completed")
                self._check_halt()
            finally:
                stream.done(reader, row)
                checkpoint = stream.checkpoints[reader]
                if checkpoint is not None:
                    self.ledger.checkpoint(batch.id, reader, checkpoint)
                in_flight.release()

        async with contextlib.AsyncExitStack() as scope:
            if self.pool is not None:
                await scope.enter_async_context(self.pool)
            iterator = stream.__aiter__()
            try:
                async for reader, row in iterator:
                    await in_flight.acquire()
                    if execution.max_rows is not None and self._taken >= execution.max_rows:
                        self._halt = f"max_rows={execution.max_rows} reached"
                    if self._halt is not None:
                        # Delivered but never run: left outstanding, so the checkpoint
                        # stays behind it and a resume picks it up first.
                        in_flight.release()
                        break
                    self._taken += 1
                    task = asyncio.create_task(one(reader, row))
                    running.add(task)
                    task.add_done_callback(running.discard)
            finally:
                await iterator.aclose()
                if running:
                    await asyncio.gather(*running)

        job.cursor = {
            str(reader): cursor.model_dump(mode="json")
            for reader, cursor in self.ledger.checkpoints(batch.id).items()
        }
        if not await asyncio.to_thread(flush, timeout=120.0):
            logger.warning("telemetry flush did not fully drain within 120s; some spans may lag")
        if self._halt is not None:
            raise BatchHalted(job, self._halt)
        return job


__all__ = ["BatchHalted", "BatchScheduler", "one_shot_agent", "sampled"]
