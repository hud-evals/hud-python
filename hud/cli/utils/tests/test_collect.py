"""``hud.cli.utils.collect`` — collecting v6 ``Variant``s from .py sources + JSON/JSONL.

The collector is what ``hud eval`` / ``hud sync`` / ``hud harbor`` use to turn a task
source into runnable ``Variant``s.
"""

from __future__ import annotations

import json
import textwrap
from typing import TYPE_CHECKING

import pytest

from hud.cli.utils.collect import collect_variants, load_variants_json
from hud.eval import Variant

if TYPE_CHECKING:
    from pathlib import Path

_ENV_PY = """\
from hud import Environment, variant

env = Environment("demo")


@env.task()
async def solve(n: int = 1):
    yield f"solve {n}"
    yield 1.0


# A module-level list of Variants (the `tasks = [...]` pattern) + a bare Variant.
tasks = [solve(n=1), solve(n=2)]
extra = solve(n=3)
"""


def _write(path: Path, content: str) -> Path:
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


# ─── collect_variants: Python sources ─────────────────────────────────


def test_collect_variants_from_py_file_picks_up_list_and_bare(tmp_path: Path) -> None:
    env_py = _write(tmp_path / "env.py", _ENV_PY)

    variants = collect_variants(str(env_py))

    assert all(isinstance(v, Variant) for v in variants)
    assert sorted(v.args["n"] for v in variants) == [1, 2, 3]  # tasks list (1,2) + bare (3)
    assert {v.task for v in variants} == {"solve"}


def test_collect_variants_from_directory_scans_py_files(tmp_path: Path) -> None:
    _write(tmp_path / "env.py", _ENV_PY)
    _write(
        tmp_path / "more.py",
        """\
        from hud import Environment

        env2 = Environment("more")

        @env2.task()
        async def ping():
            yield "ping"
            yield 1.0

        tasks = [ping()]
        """,
    )

    variants = collect_variants(str(tmp_path))

    assert {v.task for v in variants} == {"solve", "ping"}


def test_collect_variants_missing_source_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        collect_variants(str(tmp_path / "nope.py"))


# ─── load_variants_json: JSON / JSONL tasksets ────────────────────────


def test_load_variants_json_list(tmp_path: Path) -> None:
    entries = [
        {"env": {"type": "hud", "name": "demo"}, "task": "solve", "args": {"n": 1}},
        {"env": {"type": "hud", "name": "demo"}, "task": "solve", "args": {"n": 2}, "slug": "two"},
    ]
    path = _write(tmp_path / "tasks.json", json.dumps(entries))

    variants = load_variants_json(path)

    assert [v.task for v in variants] == ["solve", "solve"]
    assert [v.args["n"] for v in variants] == [1, 2]
    assert variants[1].slug == "two"


def test_load_variants_json_single_object(tmp_path: Path) -> None:
    entry = {"env": {"type": "hud", "name": "demo"}, "task": "solve", "args": {}}
    path = _write(tmp_path / "one.json", json.dumps(entry))

    variants = load_variants_json(path)

    assert len(variants) == 1
    assert variants[0].task == "solve"


def test_load_variants_jsonl(tmp_path: Path) -> None:
    lines = [
        json.dumps({"env": {"type": "url", "url": "tcp://h:7000"}, "task": "a"}),
        "",  # blank lines are skipped
        json.dumps({"env": {"type": "url", "url": "tcp://h:7000"}, "task": "b"}),
    ]
    path = _write(tmp_path / "tasks.jsonl", "\n".join(lines))

    variants = load_variants_json(path)

    assert [v.task for v in variants] == ["a", "b"]


def test_load_variants_json_rejects_scalar(tmp_path: Path) -> None:
    path = _write(tmp_path / "bad.json", "42")
    with pytest.raises(ValueError, match="expected a JSON object"):
        load_variants_json(path)


def test_load_variants_json_resolves_relative_module_ref(tmp_path: Path) -> None:
    # A ``module`` env-ref with a relative path resolves next to the taskset file,
    # so a tasks file is portable beside the env code it references.
    _write(tmp_path / "env.py", _ENV_PY)
    entry = {"env": {"type": "module", "module": "env.py", "name": "demo"}, "task": "solve"}
    path = _write(tmp_path / "tasks.jsonl", json.dumps(entry))

    variants = load_variants_json(path)

    assert len(variants) == 1
    assert variants[0].task == "solve"
