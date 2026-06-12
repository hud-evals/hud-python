"""Lightweight contract matching by robot_type and feature wiring (v0 schema).

v0 is the single-space schema: one embodiment (``robot_type``), one observation
space and one action space per contract — no ``action_modes`` /
``observation_modes`` wrappers and no ``robot_type_variables`` decision knobs.
A model that targets several embodiments or action forms ships **one contract
per form** (see spec_v0.md §5). The retired multi-mode schema is archived as
documentation under the demos ``contracts/experiments/`` corpus and is not
loadable here.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

Feature = tuple[str, dict | None]

# spec_v0 §3.4 — visual stream color-space / modality tags
IMAGE_TYPES = frozenset({"rgb", "bgr", "gray", "depth"})


def is_image_feature(feature: dict) -> bool:
    """Whether a contract feature is a visual observation stream."""
    return feature.get("type") in IMAGE_TYPES or feature.get("dtype") == "image"


def match(model: dict, robot_type: str) -> bool:
    """Whether ``model`` supports ``robot_type`` — the v0 gate, truthiness-safe.

    Support is declared solely by the model's top-level ``robot_type`` (a
    string; a list is tolerated for multi-embodiment checkpoints, see spec §3.9).
    """
    declared = model.get("robot_type")
    supported = declared if isinstance(declared, list) else [declared]
    return robot_type in supported


def split_observations(contract: dict) -> tuple[list[Feature], list[Feature]]:
    """Return (image observations, vector observations) from a contract."""
    obs = [
        (name, feat)
        for name, feat in contract.get("features", {}).items()
        if feat.get("role") == "observation"
    ]
    images = [(n, f) for n, f in obs if is_image_feature(f)]
    vectors = [(n, f) for n, f in obs if not is_image_feature(f)]
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


def _zip_features(left: list[Feature], right: list[Feature]) -> list[tuple[Feature, Feature]]:
    fill: Feature = (None, None)
    return list(itertools.zip_longest(left, right, fillvalue=fill))


def pair_observations(env: dict, model: dict) -> list[tuple[Feature, Feature]]:
    """Pair env obs -> model obs: images first, then vectors (positional within each group)."""
    env_images, env_vectors = split_observations(env)
    model_images, model_vectors = split_observations(model)
    return _zip_features(env_images, model_images) + _zip_features(env_vectors, model_vectors)


@dataclass(frozen=True)
class ActionMatch:
    signature: str
    matched: bool
    pairs: tuple[tuple[Feature, Feature], ...] = ()
    model_signature: str | None = None


def match_actions(env: dict, model: dict) -> ActionMatch:
    """Compare the env action signature against the model's, then pair features.

    v0: both sides declare exactly one action space (their top-level
    ``role == "action"`` features); a match is signature equality.
    """
    env_actions = list_actions(env)
    model_actions = list_actions(model)
    signature = action_signature(env_actions)
    model_signature = action_signature(model_actions) if model_actions else None
    if model_actions and signature == model_signature:
        pairs = tuple(_zip_features(env_actions, model_actions))
        return ActionMatch(
            signature=signature, matched=True, pairs=pairs, model_signature=model_signature
        )
    return ActionMatch(signature=signature, matched=False, model_signature=model_signature)
