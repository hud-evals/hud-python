from __future__ import annotations

import json
from typing import TYPE_CHECKING

from hud.cli.utils.project_config import (
    get_registry_id,
    get_taskset_id,
    load_project_config,
    save_project_config,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_load_reads_config_json(tmp_path: Path):
    hud_dir = tmp_path / ".hud"
    hud_dir.mkdir()
    (hud_dir / "config.json").write_text(
        json.dumps({"registryId": "reg-1", "tasksetId": "taskset-1"}),
        encoding="utf-8",
    )

    assert load_project_config(tmp_path) == {
        "registryId": "reg-1",
        "tasksetId": "taskset-1",
    }
    assert get_registry_id(tmp_path) == "reg-1"
    assert get_taskset_id(tmp_path) == "taskset-1"


def test_load_has_no_deploy_json_side_effects(tmp_path: Path):
    hud_dir = tmp_path / ".hud"
    hud_dir.mkdir()
    legacy_path = hud_dir / "deploy.json"
    legacy_path.write_text(json.dumps({"registry_id": "legacy-reg"}), encoding="utf-8")

    assert load_project_config(tmp_path) == {}
    assert legacy_path.exists()
    assert not (hud_dir / "config.json").exists()


def test_save_merges_config(tmp_path: Path):
    assert save_project_config({"registryId": "reg-1"}, tmp_path) == tmp_path / ".hud/config.json"

    changed = save_project_config({"tasksetId": "taskset-1"}, tmp_path)

    assert changed == tmp_path / ".hud/config.json"
    assert load_project_config(tmp_path) == {
        "registryId": "reg-1",
        "tasksetId": "taskset-1",
    }


def test_save_returns_none_when_unchanged(tmp_path: Path):
    save_project_config({"registryId": "reg-1"}, tmp_path)

    assert save_project_config({"registryId": "reg-1"}, tmp_path) is None
