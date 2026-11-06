"""``hud eval`` on a Batch source, and the ``hud batch`` / ``hud sync batch`` commands."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from tests.batch.conftest import write_documents
from tests.fixtures.models import Turn

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import Hud
    from tests.fixtures.models import Models
    from tests.fixtures.services import FakeServices

BATCH_PY = """\
from hud import Environment
from hud.batch import Batch, Binding, Dataset, Execution

env = Environment("docs")


@env.template()
async def scan(record_id: str):
    answer = yield f"scan {record_id}"
    yield 1.0 if answer == "ok" else 0.0


batch = Batch(
    name="docs-scan",
    template=scan,
    dataset=Dataset.file("data/*.parquet"),
    binding=Binding(row_id="id", args={"record_id": "$.id"}, where="n_chars <= 300"),
    execution=Execution(trace_sampling=0.0, rows_per_instance=None, sessions_per_instance=2),
)
"""
SCRIPTED = ("openai_compatible", "-m", "scripted")


@pytest.fixture
def gateway(models: Models, signed_in: FakeServices) -> Models:
    models.script([Turn(text="ok")])
    return models


@pytest.fixture
def batch_source(hud: Hud) -> Path:
    write_documents(hud.cwd / "data", files=2, rows_per_file=6, row_group_size=4)
    path = hud.cwd / "batch.py"
    path.write_text(BATCH_PY)
    return path


def run(hud: Hud, *argv: str) -> Any:
    """A successful ``hud`` command's ``--json`` document (``--yes`` where eval asks)."""
    flags = ("--yes", "--json") if argv[0] == "eval" else ("--json",)
    result = hud(*argv, *flags)
    assert result.exit_code == 0, result
    return result.json


def test_eval_samples_one_row_of_a_batch_by_default_and_n_with_sample(
    hud: Hud, gateway: Models, batch_source: Path
) -> None:
    one = run(hud, "eval", "batch.py", *SCRIPTED)
    three = run(hud, "eval", "batch.py", *SCRIPTED, "--sample", "3")

    assert (one["run_count"], one["mean_reward"]) == (1, 1.0)
    assert three["run_count"] == 3
    assert all(r["slug"].startswith("docs-scan-") for r in three["runs"])


def test_eval_full_runs_every_eligible_row_and_a_rerun_resumes_from_the_ledger(
    hud: Hud, gateway: Models, batch_source: Path
) -> None:
    first = run(hud, "eval", "batch.py", *SCRIPTED, "--all")
    again = run(hud, "eval", "batch.py", *SCRIPTED, "--all")

    assert (first["rows_done"], first["rows_failed"], first["completed"]) == (6, 0, 6)
    assert first["mean_reward"] == 1.0 and first["halted"] is None
    assert (hud.cwd / ".hud" / "batches").is_dir()
    assert (again["rows_done"], again["completed"]) == (0, 6)
    assert len(gateway.requests()) == 6


def test_eval_full_one_shot_makes_one_call_per_row_with_no_tools(
    hud: Hud, gateway: Models, batch_source: Path
) -> None:
    document = run(hud, "eval", "batch.py", *SCRIPTED, "--all", "--one-shot", "--instances", "2")

    assert document["rows_done"] == 6
    requests = gateway.requests()
    assert len(requests) == 6 and all(request.tools == [] for request in requests)


def test_batch_status_and_results_read_the_ledger(
    hud: Hud, gateway: Models, batch_source: Path
) -> None:
    missing = hud("batch", "status", "batch.py", "--json")
    assert missing.exit_code != 0 and "No ledger" in missing.stderr + missing.stdout

    run(hud, "eval", "batch.py", *SCRIPTED, "--all")
    status = run(hud, "batch", "status", "batch.py")
    exported = run(hud, "batch", "results", "batch.py", "-o", "results.jsonl")

    assert (status["completed"], status["error"], status["mean_reward"]) == (6, 0, 1.0)
    assert status["checkpoints"] == {"0": {"file": 1, "row_group": 0, "row": 2}}
    lines = [json.loads(line) for line in (hud.cwd / "results.jsonl").read_text().splitlines()]
    assert exported["rows"] == 6 and [row["row_id"] for row in lines] == [
        f"doc-{f}-{r}" for f in range(2) for r in range(3)
    ]


def test_sync_batch_validates_and_records_the_definition(
    hud: Hud, signed_in: FakeServices, batch_source: Path
) -> None:
    preview = run(hud, "sync", "batch", "batch.py", "--dry-run")
    saved = run(hud, "sync", "batch", "batch.py")

    assert preview["batch"]["binding"]["where"] == "n_chars <= 300"
    stored = json.loads((hud.cwd / ".hud" / "batch.json").read_text())
    assert (stored["id"], stored["env"], stored["task"]) == (saved["batch_id"], "docs", "scan")
    assert saved["files"] == 2


def test_batch_options_are_refused_for_a_taskset_source(
    hud: Hud, gateway: Models, batch_source: Path
) -> None:
    (hud.cwd / "tasks.py").write_text(
        'from hud import Environment\nenv = Environment("demo")\n'
        '@env.template()\nasync def solve(n: int = 0):\n    yield "go"\n    yield 1.0\n'
        "tasks = [solve()]\n"
    )

    result = hud("eval", "tasks.py", *SCRIPTED, "--sample", "2", "--yes", "--json")

    assert result.exit_code != 0
    assert "apply to a Batch source" in result.stderr + result.stdout
