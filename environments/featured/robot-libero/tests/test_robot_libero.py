"""Contract and grasp-label checks that need no simulator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("hud.environment.robot")
from control import grasp_sentence  # noqa: E402


@pytest.mark.parametrize("name", ["ee_abs", "ee_del"])
def test_contract_describes_one_action_space_and_two_cameras(name: str) -> None:
    contract = json.loads((ROOT / "contracts" / f"{name}.json").read_text())
    features = contract["features"]
    assert features["action"]["type"] == name
    assert features["action"]["names"][-1] == "gripper.open_close"
    assert features["observation/image"]["shape"][:2] == features["observation/wrist_image"]["shape"][:2]


def test_tasks_name_templates_of_the_env() -> None:
    tasks = json.loads((ROOT / "tasks.json").read_text())
    suites = {"libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"}
    assert {task["env"] for task in tasks} == {"robot-libero"}
    assert {task["id"] for task in tasks} == suites
    assert len({task["slug"] for task in tasks}) == len(tasks)


@pytest.mark.parametrize(
    ("opening", "close_commanded", "expected"),
    [
        (0.04, False, "open"),
        (0.0, True, "fingers met; nothing between them"),
        (0.01, True, "fingers did not meet"),
        (0.01, False, None),
        (0.003, True, "fingers did not meet"),
    ],
)
def test_grasp_sentence_labels_the_finger_opening(opening: float, close_commanded: bool, expected: str | None) -> None:
    assert grasp_sentence(opening, close_commanded=close_commanded) == expected
