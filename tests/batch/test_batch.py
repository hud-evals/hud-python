"""``Batch``: a template bound to a dataset; its identity, its rows, its samples."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel

from hud import Environment
from hud.batch import Batch, Binding, Dataset, Execution, Materialization

if TYPE_CHECKING:
    from pathlib import Path


class Found(BaseModel):
    ids: list[str]


def docs_env() -> Environment:
    env = Environment("docs")

    @env.template(returns=Found)
    async def scan(record_id: str, n_chars: int = 0):
        answer = yield record_id
        yield 1.0 if answer == record_id else 0.0

    return env


BINDING = Binding(
    row_id="id",
    args={"record_id": "$.id", "n_chars": "$.n_chars"},
    where="n_chars <= 800",
    columns={"language": "$.language"},
)


def batch_at(documents: Path, **overrides: object) -> Batch:
    env = docs_env()
    fields: dict[str, object] = {
        "name": "docs-scan",
        "template": env.tasks["scan"],
        "dataset": Dataset.file(f"{documents}/*.parquet"),
        "binding": BINDING,
    }
    fields.update(overrides)
    return Batch(**fields)  # type: ignore[arg-type]


def test_a_batch_takes_env_task_and_returns_schema_from_its_template(documents: Path) -> None:
    batch = batch_at(documents)

    assert (batch.env, batch.task) == ("docs", "scan")
    assert batch.returns_schema is not None and batch.returns_schema["required"] == ["ids"]
    assert batch.template is not None
    assert batch.id.startswith("docs-scan-") and len(batch.id) == len("docs-scan-") + 12


def test_the_id_follows_what_rows_the_batch_mints_not_how_it_runs(documents: Path) -> None:
    base = batch_at(documents)

    same_rows = batch_at(
        documents,
        execution=Execution(mode="one_shot", instances=8),
        materialization=Materialization(parallel_readers=4, prefetch_rows=10),
    )
    other_rows = batch_at(documents, binding=BINDING.model_copy(update={"where": None}))
    other_shards = batch_at(documents, materialization=Materialization(shards=4))

    assert same_rows.id == base.id
    assert other_rows.id != base.id and other_shards.id != base.id


def test_a_binding_that_disagrees_with_the_template_or_dataset_is_rejected(
    documents: Path,
) -> None:
    with pytest.raises(ValueError, match="missing required args"):
        batch_at(documents, binding=Binding(row_id="id", args={}))
    with pytest.raises(ValueError, match="columns the dataset does not select"):
        batch_at(documents, dataset=Dataset.file(f"{documents}/*.parquet", columns=["id"]))
    with pytest.raises(ValueError, match="env/task disagree"):
        batch_at(documents, env="other")


def test_a_stored_batch_rebuilds_without_its_template(documents: Path) -> None:
    stored = batch_at(documents).model_dump(mode="json")

    rebuilt = Batch.model_validate(stored)

    assert rebuilt.template is None
    assert rebuilt.id == batch_at(documents).id
    task = rebuilt.bind({"id": "x", "n_chars": 3, "language": "go"})
    assert (task.slug, task.args, task.columns) == (
        f"{rebuilt.id}:x",
        {"record_id": "x", "n_chars": 3},
        {"language": "go"},
    )


async def test_sample_is_a_taskset_of_eligible_rows_deterministic_per_seed(
    documents: Path,
) -> None:
    batch = batch_at(documents)

    sample = await batch.sample(5, seed=7)
    again = await batch.sample(5, seed=7)
    everything = await batch.sample(1000)

    assert len(sample) == 5 and [t.slug for t in sample] == [t.slug for t in again]
    assert all(task.args["n_chars"] <= 800 for task in sample)
    assert len(everything) == 24  # 8 eligible rows x 3 files
    assert next(iter(everything)).slug == f"{batch.id}:doc-0-0"
    with pytest.raises(ValueError, match="sample size"):
        await batch.sample(0)


def test_from_file_finds_the_batch_a_source_defines(documents: Path, tmp_path: Path) -> None:
    source = tmp_path / "batch.py"
    source.write_text(
        f"""
from pydantic import BaseModel
from hud import Environment
from hud.batch import Batch, Binding, Dataset

env = Environment("docs")

@env.template()
async def scan(record_id: str):
    yield record_id
    yield 1.0

batch = Batch(
    name="from-source",
    template=scan,
    dataset=Dataset.file("{documents}/*.parquet"),
    binding=Binding(row_id="id", args={{"record_id": "$.id"}}),
)
"""
    )

    found = Batch.from_file(source)

    assert found.name == "from-source" and found.template is not None
    with pytest.raises(ValueError, match="no Batch named 'other'"):
        Batch.from_file(source, name="other")
