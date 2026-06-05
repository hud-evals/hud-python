"""``hud sync`` core: local specs, diff signatures, column inference, upload/export.

Covers the offline pieces that drive sync's create/update/skip diff against the
platform; network calls (``httpx`` / ``fetch_remote_tasks``) are mocked.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest
import typer

from hud.cli import sync as sync_mod
from hud.cli.sync import (
    _build_column_definitions,
    _build_local_specs,
    _compute_remote_signature,
    _compute_signature,
    _diff_and_display,
    _export_remote_tasks,
    _infer_column_type,
    _upload_tasks,
)
from hud.environment import Environment
from hud.eval import variant
from hud.utils.hud_console import HUDConsole

if TYPE_CHECKING:
    from pathlib import Path

_console = HUDConsole()


def _env() -> Environment:
    return Environment("demo")


# ─── _build_local_specs ───────────────────────────────────────────────


def test_build_local_specs_defaults_slug_and_prefixes_scenario() -> None:
    specs = _build_local_specs([variant(_env(), "solve", n=1)], _console)

    assert len(specs) == 1
    spec = specs[0]
    assert spec["scenario_name"] == "demo:solve"  # env-prefixed
    assert spec["args"] == {"n": 1}
    assert spec["slug"].startswith("solve-")  # default_slug = task + args hash
    assert spec["validation"] is None
    assert spec["agent_config"] is None
    assert spec["columns"] is None


def test_build_local_specs_threads_explicit_metadata() -> None:
    v = variant(
        _env(),
        "solve",
        slug="custom-slug",
        validation=[{"name": "submit", "arguments": {"answer": "x"}}],
        agent_config={"system_prompt": "be precise"},
        columns={"tier": "hard"},
        n=2,
    )

    spec = _build_local_specs([v], _console)[0]

    assert spec["slug"] == "custom-slug"
    assert spec["validation"] == [{"name": "submit", "arguments": {"answer": "x"}}]
    assert spec["agent_config"] == {"system_prompt": "be precise"}
    assert spec["columns"] == {"tier": "hard"}


def test_build_local_specs_rejects_duplicate_slugs() -> None:
    env = _env()
    dupes = [variant(env, "solve", slug="same"), variant(env, "solve", slug="same", n=9)]
    with pytest.raises(typer.Exit):
        _build_local_specs(dupes, _console)


def test_build_local_specs_skips_non_variant_items() -> None:
    specs = _build_local_specs([object(), variant(_env(), "solve")], _console)
    assert len(specs) == 1
    assert specs[0]["scenario_name"] == "demo:solve"


# ─── signatures (diff identity) ───────────────────────────────────────


def test_signature_ignores_env_prefix() -> None:
    args: dict[str, Any] = {"n": 1}
    assert _compute_signature("demo:solve", args, None, None) == _compute_signature(
        "other-env:solve", args, None, None
    )


def test_signature_changes_with_args_and_metadata() -> None:
    base = _compute_signature("solve", {"n": 1}, None, None)
    assert base != _compute_signature("solve", {"n": 2}, None, None)
    assert base != _compute_signature("solve", {"n": 1}, [{"name": "submit"}], None)
    assert base != _compute_signature("solve", {"n": 1}, None, {"system_prompt": "x"})


def test_local_and_remote_signatures_match_for_same_task() -> None:
    v = variant(
        _env(),
        "solve",
        validation=[{"name": "submit"}],
        agent_config={"system_prompt": "p"},
        columns={"tier": "easy"},
        n=1,
    )
    spec = _build_local_specs([v], _console)[0]

    # A platform task carrying the same logical content must produce the same
    # signature, so the diff sees it as "unchanged" rather than create+delete.
    remote_task = {
        "scenario": spec["scenario_name"],
        "args": spec["args"],
        "validation": spec["validation"],
        "agent_config": spec["agent_config"],
        "column_values": spec["columns"],
    }
    assert _compute_remote_signature(remote_task) == spec["signature"]


# ─── column inference ─────────────────────────────────────────────────


def test_infer_column_type() -> None:
    assert _infer_column_type([]) == "text"
    assert _infer_column_type([1, 2.0, None]) == "number"
    assert _infer_column_type([["a"], ["b", "c"]]) == "multi-select"
    assert _infer_column_type(["easy", "hard"]) == "text"
    assert _infer_column_type([1, "x"]) == "text"  # mixed -> text


def test_build_column_definitions_infers_types() -> None:
    specs = [
        {"columns": {"difficulty": 1, "tags": ["a", "b"]}},
        {"columns": {"difficulty": 2, "tags": ["b", "c"]}},
    ]
    defs = _build_column_definitions(specs)
    assert defs is not None
    assert defs["difficulty"]["type"] == "number"
    assert defs["tags"]["type"] == "multi-select"
    assert defs["tags"]["options"] == ["a", "b", "c"]


def test_build_column_definitions_none_without_columns() -> None:
    assert _build_column_definitions([{"slug": "x"}]) is None


# ─── diff ─────────────────────────────────────────────────────────────


def test_diff_classifies_create_update_unchanged() -> None:
    env = _env()
    specs = _build_local_specs(
        [
            variant(env, "a", slug="a"),
            variant(env, "b", slug="b"),
            variant(env, "c", slug="c"),
        ],
        _console,
    )
    by_slug = {s["slug"]: s for s in specs}
    remote = [
        {"slug": "a", "scenario": by_slug["a"]["scenario_name"], "args": {}},  # unchanged
        {"slug": "b", "scenario": "demo:b", "args": {"changed": 1}},  # update (sig differs)
        {"slug": "old", "scenario": "demo:old", "args": {}},  # remote-only
    ]
    # "c" is local-only -> create
    to_upload = _diff_and_display(specs, remote, "demo", "tid", True, _console)

    slugs = {s["slug"] for s in to_upload}
    assert "c" in slugs  # created
    assert "b" in slugs  # updated
    assert "a" not in slugs  # unchanged, not re-uploaded


# ─── upload (mock httpx) ──────────────────────────────────────────────


def test_upload_tasks_posts_expected_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, *, json: Any, headers: Any, timeout: float) -> Any:
        captured["url"] = url
        captured["json"] = json
        return MagicMock(raise_for_status=lambda: None, json=lambda: {"ok": True})

    monkeypatch.setattr(sync_mod.httpx, "post", fake_post)

    specs = _build_local_specs(
        [variant(_env(), "solve", slug="s1", validation=[{"name": "submit"}], n=1)],
        _console,
    )
    result = _upload_tasks(specs, "demo", "https://api", {"Authorization": "Bearer x"})

    assert result == {"ok": True}
    assert captured["url"].endswith("/tasks/upload")
    task = captured["json"]["tasks"][0]
    assert task["slug"] == "s1"
    assert task["scenario"] == "demo:solve"
    assert task["validation"] == [{"name": "submit"}]


# ─── export (mock fetch) ──────────────────────────────────────────────


def test_export_remote_tasks_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tasks = [{"slug": "a", "scenario": "demo:a", "args": {"n": 1}}]
    monkeypatch.setattr(sync_mod, "fetch_remote_tasks", lambda *_a, **_k: tasks)
    out = tmp_path / "tasks.json"

    _export_remote_tasks("tid", "demo", str(out), "https://api", {}, _console)

    assert json.loads(out.read_text(encoding="utf-8"))[0]["slug"] == "a"


def test_export_remote_tasks_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tasks = [{"slug": "a", "scenario": "demo:a", "args": {"n": 1}, "env": {"name": "demo"}}]
    monkeypatch.setattr(sync_mod, "fetch_remote_tasks", lambda *_a, **_k: tasks)
    out = tmp_path / "tasks.csv"

    _export_remote_tasks("tid", "demo", str(out), "https://api", {}, _console)

    header = out.read_text(encoding="utf-8").splitlines()[0]
    assert "slug" in header
    assert "arg:n" in header


def test_export_remote_tasks_bad_suffix_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sync_mod, "fetch_remote_tasks", lambda *_a, **_k: [{"slug": "a"}])
    bad = str(tmp_path / "tasks.txt")
    with pytest.raises(typer.Exit):
        _export_remote_tasks("tid", "demo", bad, "https://api", {}, _console)
