"""Lightweight contract matching by robot_type and feature wiring.

NOTE (In Development): the `action_modes` (see `model_action_modes`) and
`observation_modes` (see `model_features`) handling below targets the *experimental*
multi-mode contract schema (specs in the demos `contracts/experiments/` corpus). The
going-forward **standard** schema is one action space and one observation space per
contract (no `*_modes` wrappers); see §5 of the SPEC.md co-located in this package.
This matcher has **not** been
updated to that standard — it still centers on the experimental wrappers, so the
standard split contracts do not exercise these code paths (top-level `action.*`
features only fall back through `model_action_modes`'s `default` branch). Treat this
as in-development until the design settles."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

Feature = tuple[str, dict | None]


def match(model: dict, robot_type: str) -> dict | None:
    """Decision variables for ``robot_type``, or None if the model does not support it."""
    return model.get("robot_type_variables", {}).get(robot_type)


def model_features(model: dict, robot_type: str | None = None) -> dict:
    """Model features for pairing; swaps obs state when ``observation_mode`` is set."""
    features = dict(model.get("features", {}))
    if not robot_type:
        return features
    mode_name = model.get("robot_type_variables", {}).get(robot_type, {}).get("observation_mode")
    if not mode_name:
        return features
    mode_feats = model.get("observation_modes", {}).get(mode_name, {}).get("features", {})
    features = {k: v for k, v in features.items() if not k.startswith("observation.state.")}
    features.update(mode_feats)
    return features


def _contract_with_features(contract: dict, features: dict) -> dict:
    return {**contract, "features": features}


def _is_image(feature: dict) -> bool:
    return feature.get("type") == "rgb" or feature.get("dtype") == "image"


def split_observations(contract: dict) -> tuple[list[Feature], list[Feature]]:
    """Return (image observations, vector observations) from a contract."""
    obs = [
        (name, feat)
        for name, feat in contract.get("features", {}).items()
        if feat.get("role") == "observation"
    ]
    images = [(n, f) for n, f in obs if _is_image(f)]
    vectors = [(n, f) for n, f in obs if not _is_image(f)]
    return images, vectors


def list_actions(contract: dict) -> list[Feature]:
    """Action features sorted by ``order``."""
    actions = (
        (name, feat)
        for name, feat in contract.get("features", {}).items()
        if feat.get("role") == "action"
    )
    return sorted(actions, key=lambda item: item[1].get("order", item[0]))


def action_signature(features: list[Feature]) -> str:
    """Chain of ``state_type`` values, e.g. ``EE_DEL_POS+EE_DEL_ROT+GRIPPER_ABS_POS``."""
    return "+".join(feat.get("state_type", feat.get("type", "?")) for _, feat in features)


def model_action_modes(model: dict, robot_type: str | None = None) -> dict[str, dict]:
    """Map action signature -> {mode, features}. Top-level actions register as ``default``."""
    modes: dict[str, dict] = {}
    for mode_name, mode in model.get("action_modes", {}).items():
        feats = sorted(mode.get("features", {}).items(), key=lambda x: x[1].get("order", x[0]))
        modes[action_signature(feats)] = {"mode": mode_name, "features": feats}
    actions = list_actions(model)
    if actions:
        modes.setdefault(action_signature(actions), {"mode": "default", "features": actions})
    if robot_type:
        adapter = model.get("robot_type_variables", {}).get(robot_type, {}).get("action_adapter")
        if adapter and adapter in model.get("action_modes", {}):
            feats = sorted(
                model["action_modes"][adapter]["features"].items(),
                key=lambda x: x[1].get("order", x[0]),
            )
            modes[action_signature(feats)] = {"mode": adapter, "features": feats}
    return modes


def _zip_features(left: list[Feature], right: list[Feature]) -> list[tuple[Feature, Feature]]:
    fill: Feature = (None, None)
    return list(itertools.zip_longest(left, right, fillvalue=fill))


def pair_observations(
    env: dict, model: dict, robot_type: str | None = None
) -> list[tuple[Feature, Feature]]:
    """Pair env obs -> model obs: images first, then vectors (positional within each group)."""
    model_view = _contract_with_features(model, model_features(model, robot_type))
    env_images, env_vectors = split_observations(env)
    model_images, model_vectors = split_observations(model_view)
    return _zip_features(env_images, model_images) + _zip_features(env_vectors, model_vectors)


@dataclass(frozen=True)
class ActionMatch:
    signature: str
    matched: bool
    mode: str | None = None
    pairs: tuple[tuple[Feature, Feature], ...] = ()
    available_signatures: tuple[str, ...] = ()


def match_actions(env: dict, model: dict, robot_type: str | None = None) -> ActionMatch:
    """Select a model action mode whose signature matches the env, then pair features."""
    env_actions = list_actions(env)
    signature = action_signature(env_actions)
    modes = model_action_modes(model, robot_type)
    if signature in modes:
        selected = modes[signature]
        pairs = tuple(_zip_features(env_actions, selected["features"]))
        return ActionMatch(signature=signature, matched=True, mode=selected["mode"], pairs=pairs)
    return ActionMatch(
        signature=signature,
        matched=False,
        available_signatures=tuple(sorted(modes)),
    )
