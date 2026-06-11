"""Contract matcher tests against the v0 single-space schema.

v0 is one embodiment (``robot_type``) and one action space + one observation
space per contract: actions match on signature equality between the two sides'
top-level ``role == "action"`` features, and observations pair positionally
(images first, then vectors). The inline fixtures below are written in that
single-space style; the ``fixtures/`` pair (libero env / pi05_libero model) is
a known-MATCH real-world pair in the same style.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hud.environment.robots.contracts import (
    action_signature,
    integration_review,
    list_actions,
    match,
    match_actions,
    pair_observations,
    render_match,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ── inline single-space fixtures ──────────────────────────────────────────────


def make_env_contract(**overrides: Any) -> dict[str, Any]:
    contract = {
        "robot_type": "bot_x",
        "control_rate": 10,
        "features": {
            "observation.images.cam": {
                "role": "observation",
                "type": "rgb",
                "dtype": "uint8",
                "shape": [64, 64, 3],
            },
            "observation.state.eef_pos": {
                "role": "observation",
                "state_type": "EE_ABS_POS",
                "dtype": "float32",
                "shape": [3],
                "order": "0-2",
            },
            "action.delta_eef_pos": {
                "role": "action",
                "state_type": "EE_DEL_POS",
                "dtype": "float32",
                "shape": [3],
                "order": "0-2",
            },
            "action.gripper": {
                "role": "action",
                "state_type": "GRIPPER_ABS_POS",
                "dtype": "float32",
                "shape": [1],
                "order": "3",
            },
        },
    }
    contract.update(overrides)
    return contract


def make_model_contract(**overrides: Any) -> dict[str, Any]:
    contract = {
        "model": "stub_policy",
        "robot_type": "bot_x",
        "control_rate": 10,
        "features": {
            "observation.images.image": {
                "role": "observation",
                "type": "rgb",
                "dtype": "uint8",
                "shape": [64, 64, 3],
            },
            "observation.state.eef_pos": {
                "role": "observation",
                "state_type": "EE_ABS_POS",
                "dtype": "float32",
                "shape": [3],
                "order": "0-2",
            },
            "action.delta_eef_pos": {
                "role": "action",
                "state_type": "EE_DEL_POS",
                "dtype": "float32",
                "shape": [3],
                "order": "0-2",
            },
            "action.gripper": {
                "role": "action",
                "state_type": "GRIPPER_ABS_POS",
                "dtype": "float32",
                "shape": [1],
                "order": "3",
            },
        },
    }
    contract.update(overrides)
    return contract


# ── match(): robot_type gating ────────────────────────────────────────────────


def test_match_gates_on_robot_type() -> None:
    # v0: support is the top-level robot_type; match is a plain truthy bool.
    model = make_model_contract()
    assert match(model, "bot_x") is True
    assert match(model, "other_bot") is False  # unsupported


def test_match_gates_on_robot_type_list() -> None:
    # A list robot_type is tolerated for multi-embodiment checkpoints (spec §3.9).
    model = make_model_contract(robot_type=["bot_x", "bot_y"])
    assert match(model, "bot_y") is True
    assert match(model, "bot_z") is False


# ── pair_observations(): positional image/vector pairing ─────────────────────


def test_pair_observations_pairs_images_then_vectors_positionally() -> None:
    env, model = make_env_contract(), make_model_contract()
    pairs = pair_observations(env, model)
    assert len(pairs) == 2
    (env_img, model_img), (env_vec, model_vec) = pairs
    assert env_img[0] == "observation.images.cam"
    assert model_img[0] == "observation.images.image"
    assert env_vec[0] == "observation.state.eef_pos"
    assert model_vec[0] == "observation.state.eef_pos"


def test_pair_observations_fills_missing_side_with_none() -> None:
    env = make_env_contract()
    # Model with an extra (second) image slot: env side of that pair is (None, None).
    model = make_model_contract()
    model["features"]["observation.images.wrist"] = {
        "role": "observation",
        "type": "rgb",
        "dtype": "uint8",
        "shape": [64, 64, 3],
    }
    pairs = pair_observations(env, model)
    img_pairs = [p for p in pairs if p[1][1] and p[1][1].get("type") == "rgb"]
    assert len(img_pairs) == 2
    unmatched = img_pairs[1]
    assert unmatched[0] == (None, None)
    assert unmatched[1][0] == "observation.images.wrist"


# ── match_actions(): signature equality between the two single spaces ─────────


def test_match_actions_matches_on_signature_equality() -> None:
    env, model = make_env_contract(), make_model_contract()
    result = match_actions(env, model)
    assert result.matched is True
    assert result.signature == "EE_DEL_POS+GRIPPER_ABS_POS"
    assert result.model_signature == result.signature
    assert len(result.pairs) == 2
    assert result.pairs[0][0][0] == "action.delta_eef_pos"
    assert result.pairs[0][1][0] == "action.delta_eef_pos"


def test_match_actions_signature_mismatch() -> None:
    env = make_env_contract()
    env["features"]["action.delta_eef_pos"]["state_type"] = "JOINT_DEL_POS"
    result = match_actions(env, make_model_contract())
    assert result.matched is False
    assert result.signature == "JOINT_DEL_POS+GRIPPER_ABS_POS"
    assert result.model_signature == "EE_DEL_POS+GRIPPER_ABS_POS"


def test_action_signature_sorted_by_order() -> None:
    env = make_env_contract()
    actions = list_actions(env)
    assert [name for name, _ in actions] == ["action.delta_eef_pos", "action.gripper"]
    assert action_signature(actions) == "EE_DEL_POS+GRIPPER_ABS_POS"


# ── integration_review(): gap detection ───────────────────────────────────────


def test_integration_review_clean_match_has_no_problems() -> None:
    review = integration_review(make_env_contract(), make_model_contract())
    assert review is not None
    assert review.problems == []


def test_integration_review_returns_none_when_robot_type_unsupported() -> None:
    model = make_model_contract(robot_type="other_bot")
    assert integration_review(make_env_contract(), model) is None


def test_integration_review_detects_shape_mismatch() -> None:
    model = make_model_contract()
    model["features"]["observation.state.eef_pos"]["shape"] = [6]
    review = integration_review(make_env_contract(), model)
    assert review is not None
    shape_gaps = [g for g in review.problems if "shape mismatch" in g.issue]
    assert len(shape_gaps) == 1
    assert shape_gaps[0].category == "obs"
    assert "env.shape=[3] vs model.shape=[6]" in shape_gaps[0].spec


def test_integration_review_detects_control_rate_mismatch() -> None:
    review = integration_review(make_env_contract(), make_model_contract(control_rate=30))
    assert review is not None
    rate_gaps = [g for g in review.problems if g.issue == "control_rate mismatch"]
    assert len(rate_gaps) == 1
    assert rate_gaps[0].category == "global"
    assert "env.control_rate=10 vs model.control_rate=30" in rate_gaps[0].spec


def test_integration_review_reports_unmatched_action_signature() -> None:
    env = make_env_contract()
    env["features"]["action.gripper"]["state_type"] = "GRIPPER_DEL_POS"
    review = integration_review(env, make_model_contract())
    assert review is not None
    act_gaps = [g for g in review.problems if g.category == "act"]
    assert any("action signature mismatch" in g.issue for g in act_gaps)


# ── render_match(): terminal rendering ────────────────────────────────────────


def test_render_match_reports_match() -> None:
    out = render_match(make_model_contract(), make_env_contract())
    assert isinstance(out, str)
    assert "MATCH" in out
    assert "NO MATCH" not in out
    assert "[EE_DEL_POS+GRIPPER_ABS_POS]" in out


def test_render_match_reports_no_match_for_unknown_robot_type() -> None:
    env = make_env_contract(robot_type="alien_bot")
    out = render_match(make_model_contract(), env)
    assert "NO MATCH" in out
    assert "bot_x" in out  # lists the model's supported robots


def test_render_match_includes_integration_review_when_requested() -> None:
    model = make_model_contract(control_rate=30)
    out = render_match(model, make_env_contract(), integration=True)
    assert "integration review" in out
    assert "control_rate mismatch" in out


# ── real-world fixtures: libero env <-> pi05_libero model ────────────────────


@pytest.fixture(scope="module")
def libero_env() -> dict[str, Any]:
    return json.loads((FIXTURES / "libero.json").read_text())


@pytest.fixture(scope="module")
def pi05_model() -> dict[str, Any]:
    return json.loads((FIXTURES / "pi05_libero.json").read_text())


def test_libero_pi05_pair_matches(libero_env: dict, pi05_model: dict) -> None:
    assert match(pi05_model, libero_env["robot_type"])
    action = match_actions(libero_env, pi05_model)
    assert action.matched is True
    out = render_match(pi05_model, libero_env, integration=True)
    assert "MATCH" in out
    assert "NO MATCH" not in out


def test_libero_pi05_review_has_only_known_gaps(libero_env: dict, pi05_model: dict) -> None:
    review = integration_review(libero_env, pi05_model)
    assert review is not None
    # The known wiring difference is the image dtype (env uint8 vs model float32);
    # there must be no action-side or control-rate gaps.
    assert all(g.category != "act" for g in review.problems)
    assert all(g.issue != "control_rate mismatch" for g in review.problems)
