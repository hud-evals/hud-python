"""``BatchScheduler``: every eligible row once, through warm instances, resumable by ledger."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from hud import Environment
from hud.agents import OpenAIChatAgent
from hud.agents.types import OpenAIChatConfig
from hud.batch import Batch, BatchHalted, Binding, Dataset, Execution, Ledger, Materialization
from hud.batch.scheduler import one_shot_agent, sampled
from hud.eval import LocalRuntime, Task
from tests.fixtures.agents import ScriptedAgent
from tests.fixtures.models import Turn

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import HudEnv
    from tests.fixtures.models import Models
    from tests.fixtures.services import FakeServices


def docs_env() -> Environment:
    env = Environment("docs")

    @env.template()
    async def scan(record_id: str):
        answer = yield record_id
        if answer == "boom":
            raise RuntimeError("grader exploded")
        yield 1.0 if answer == record_id else 0.0

    return env


def batch_at(
    documents: Path,
    *,
    execution: Execution | None = None,
    materialization: Materialization | None = None,
    where: str | None = None,
) -> Batch:
    env = docs_env()
    return Batch(
        name="scan",
        template=env.tasks["scan"],
        dataset=Dataset.file(f"{documents}/*.parquet", columns=["id", "n_chars"]),
        binding=Binding(row_id="id", args={"record_id": "$.id"}, where=where),
        execution=execution or Execution(trace_sampling=0.0, rows_per_instance=None),
        materialization=materialization or Materialization(),
    )


def echo() -> ScriptedAgent:
    return ScriptedAgent(lambda prompt: prompt)


async def test_every_eligible_row_runs_once_and_lands_in_the_ledger(documents: Path) -> None:
    batch = batch_at(documents, where="n_chars <= 500")
    ledger = Ledger()

    job = await batch.run(echo(), ledger=ledger)

    results = list(ledger.results(batch.id))
    assert (job.batch_id, job.rows_done, job.rows_failed, job.runs) == (batch.id, 15, 0, [])
    assert [r.row_id for r in results] == [f"doc-{f}-{r}" for f in range(3) for r in range(5)]
    assert {r.reward for r in results} == {1.0} and all(not r.reported for r in results)
    assert ledger.summary(batch.id) == {
        "completed": 15,
        "error": 0,
        "cancelled": 0,
        "mean_reward": 1.0,
    }
    assert job.cursor == {"0": {"file": 2, "row_group": 1, "row": 0}}


async def test_failed_rows_are_recorded_with_their_trace_and_retried_up_to_retries(
    documents: Path,
) -> None:
    batch = batch_at(
        documents,
        where="n_chars <= 200",
        execution=Execution(
            trace_sampling=0.0, rows_per_instance=None, retries=1, sessions_per_instance=1
        ),
    )
    ledger = Ledger()
    answers = iter(["boom", "boom", "boom", "doc-0-1"])

    class Flaky(ScriptedAgent):
        def __init__(self) -> None:
            super().__init__(lambda prompt: next(answers, prompt))

    job = await batch.run(Flaky(), ledger=ledger)

    results = {r.row_id: r for r in ledger.results(batch.id)}
    assert (job.rows_done, job.rows_failed) == (5, 1)
    first = results["doc-0-0"]
    assert (first.status, first.attempt, first.reward) == ("error", 2, None)
    assert first.error is not None and "grader exploded" in first.error
    assert first.trace is not None and first.trace["status"] == "error"
    assert (results["doc-0-1"].status, results["doc-0-1"].attempt) == ("completed", 2)


async def test_a_restart_with_the_same_ledger_finishes_the_rest_without_repeats(
    documents: Path,
) -> None:
    ledger = Ledger()
    batch = batch_at(
        documents,
        execution=Execution(trace_sampling=0.0, rows_per_instance=None, max_rows=11),
    )

    with pytest.raises(BatchHalted, match="max_rows=11"):
        await batch.run(echo(), ledger=ledger)
    assert ledger.summary(batch.id)["completed"] == 11
    assert ledger.checkpoints(batch.id)[0].key() < (1, 1, 0)

    agent = echo()
    job = await batch_at(documents).run(agent, ledger=ledger)

    assert (job.rows_done, len(agent.prompts)) == (19, 19)
    assert sorted(r.row_id for r in ledger.results(batch.id)) == sorted(
        f"doc-{f}-{r}" for f in range(3) for r in range(10)
    )


async def test_parallel_readers_and_a_pool_of_instances_cover_the_shard(documents: Path) -> None:
    boots = []

    def build(task: Task) -> Environment:
        boots.append(task.slug)
        return docs_env()

    batch = batch_at(
        documents,
        execution=Execution(
            trace_sampling=0.0, instances=2, sessions_per_instance=2, rows_per_instance=10
        ),
        materialization=Materialization(parallel_readers=3, prefetch_rows=4, shards=2, shard=1),
    )
    ledger = Ledger()

    job = await batch.run(echo(), runtime=LocalRuntime(build), ledger=ledger)

    rows = list(ledger.results(batch.id))
    # Shard 1 of 2 over 3 readers = row-group ordinals 3, 4, 5 (file 1): 4 + 4 + 2 rows.
    assert job.rows_done == len(rows) == 10
    assert {r.reader for r in rows} == {0, 1, 2}
    assert 1 <= len(boots) <= 2
    assert set(job.cursor or {}) == {"0", "1", "2"}


async def test_the_error_rate_breaker_halts_after_the_minimum_window(documents: Path) -> None:
    batch = batch_at(
        documents,
        execution=Execution(trace_sampling=0.0, rows_per_instance=None, stop_on_error_rate=0.5),
    )
    ledger = Ledger()

    with pytest.raises(BatchHalted, match="error rate") as halted:
        await batch.run(ScriptedAgent("boom"), ledger=ledger)

    # The breaker trips at the 20th finished row; up to width - 1 more were already running.
    assert 20 <= halted.value.job.rows_failed <= 23
    assert ledger.summary(batch.id)["error"] == halted.value.job.rows_failed


async def test_sampled_rows_report_traces_and_the_rest_stay_local(
    documents: Path, reporting: FakeServices
) -> None:
    batch = batch_at(
        documents,
        where="n_chars <= 300",
        execution=Execution(trace_sampling=0.5, rows_per_instance=None),
    )
    ledger = Ledger()

    await batch.run(echo(), ledger=ledger)

    results = list(ledger.results(batch.id))
    reported = [r for r in results if r.reported]
    assert 0 < len(reported) < len(results)
    assert all(sampled(r.slug, 0.5) == r.reported for r in results)
    entered = reporting.requests("api", "POST", "/v2/trace/{id}/enter")
    assert {r.path.split("/")[3] for r in entered} == {r.trace_id for r in reported}
    assert sampled("x", 1.0) and not sampled("x", 0.0)


async def test_one_shot_mode_reconfigures_the_agent_and_answers_in_one_call(
    documents: Path, models: Models, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_API_KEY="k")
    ids = [f"doc-{f}-{r}" for f in range(3) for r in range(2)]
    models.script([Turn(text=id) for id in ids])
    batch = batch_at(
        documents,
        where="n_chars <= 200",
        execution=Execution(mode="one_shot", trace_sampling=0.0, rows_per_instance=None),
    )
    ledger = Ledger()

    job = await batch.run(OpenAIChatAgent(OpenAIChatConfig(model="qwen3.6-plus")), ledger=ledger)

    assert job.rows_done == 6
    assert all(request.tools == [] for request in models.requests())
    with pytest.raises(TypeError, match="needs a provider tool agent"):
        one_shot_agent(echo(), None)


async def test_a_stored_batch_needs_an_explicit_placement(documents: Path) -> None:
    stored = Batch.model_validate(batch_at(documents).model_dump(mode="json"))

    with pytest.raises(ValueError, match="no placement"):
        await stored.run(echo())

    job = await stored.run(echo(), runtime=LocalRuntime(docs_env()), ledger=Ledger())
    assert job.rows_done == 30


def test_the_ledger_exports_results(documents: Path, tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    from hud.batch import Cursor, RowResult

    ledger = Ledger(tmp_path / "ledger.sqlite")
    for i in range(3):
        ledger.record(
            RowResult(
                batch_id="b",
                row_id=f"r{i}",
                slug=f"b:r{i}",
                attempt=1,
                status="completed",
                reward=0.5,
                cursor=Cursor(file=0, row_group=0, row=i),
                finished_at="now",
            )
        )
    ledger.record(
        RowResult(
            batch_id="b",
            row_id="r1",
            slug="b:r1",
            attempt=2,
            status="error",
            reward=None,
            error="x",
            cursor=Cursor(file=0, row_group=0, row=1),
            finished_at="later",
            trace={"steps": []},
        )
    )

    jsonl = ledger.export("b", tmp_path / "out.jsonl").read_text().splitlines()
    parquet = pq.read_table(ledger.export("b", tmp_path / "out.parquet"))

    assert len(jsonl) == 3 and parquet.num_rows == 3  # the retry superseded r1
    superseded = ledger.get("b:r1")
    assert superseded is not None and superseded.attempt == 2
    assert ledger.summary("b") == {"completed": 2, "error": 1, "cancelled": 0, "mean_reward": 0.5}
    with pytest.raises(ValueError, match="unsupported export format"):
        ledger.export("b", tmp_path / "out.csv")
