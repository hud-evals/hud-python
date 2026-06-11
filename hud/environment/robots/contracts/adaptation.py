"""Integration gap analysis between matched env/model feature pairs."""

from __future__ import annotations

from dataclasses import dataclass, field

from .matching import match, match_actions, pair_observations


def _short(name: str | None) -> str:
    if not name:
        return "(none)"
    return name.rsplit(".", 1)[-1]


def _is_image(feature: dict) -> bool:
    return feature.get("type") == "rgb" or feature.get("dtype") == "image"


def _pair_label(env_name: str | None, model_name: str | None) -> str:
    return f"{_short(env_name)} → {_short(model_name)}"


@dataclass(frozen=True)
class Gap:
    """One detected mismatch with the spec fields that triggered it."""

    category: str  # img | obs | act | global
    issue: str
    spec: str  # e.g. "env.dtype=uint8 vs model.dtype=float32"


@dataclass
class IntegrationReview:
    """Structured integration review for a robot_type match."""

    scope: list[str] = field(default_factory=list)
    problems: list[Gap] = field(default_factory=list)


def _compare_feature_pair(
    env_name: str | None,
    env_f: dict | None,
    model_name: str | None,
    model_f: dict | None,
    *,
    category: str,
) -> list[Gap]:
    """Compare one env↔model feature pair."""
    gaps: list[Gap] = []
    label = _pair_label(env_name, model_name)

    if env_f is None and model_f is None:
        return gaps

    if env_f is None and model_f is not None:
        if model_f.get("padding"):
            return gaps
        gaps.append(
            Gap(
                category,
                f"{label}: model expects input, env has no source",
                f"model.shape={model_f.get('shape')}",
            )
        )
        return gaps

    if env_f is not None and model_f is None:
        gaps.append(
            Gap(
                category,
                f"{label}: env emits feature, model has no slot",
                f"env.state_type={env_f.get('state_type', env_f.get('type'))}",
            )
        )
        return gaps

    assert env_f is not None and model_f is not None
    if model_f.get("padding"):
        return gaps

    if _is_image(env_f):
        env_dtype, model_dtype = env_f.get("dtype"), model_f.get("dtype")
        if env_dtype != model_dtype:
            gaps.append(
                Gap(
                    category,
                    f"{label}: dtype mismatch",
                    f"env.dtype={env_dtype} vs model.dtype={model_dtype}",
                )
            )
        env_shape, model_shape = env_f.get("shape"), model_f.get("shape")
        if env_shape != model_shape:
            gaps.append(
                Gap(
                    category,
                    f"{label}: shape mismatch",
                    f"env.shape={env_shape} vs model.shape={model_shape}",
                )
            )
        env_layout = env_f.get("state_representation")
        model_layout = model_f.get("state_representation")
        if env_layout and model_layout and env_layout != model_layout:
            gaps.append(
                Gap(
                    category,
                    f"{label}: layout mismatch",
                    f"env.state_representation={env_layout} "
                    f"vs model.state_representation={model_layout}",
                )
            )
        return gaps

    if env_f.get("type") == "language" or model_f.get("type") == "language":
        return gaps

    env_st, model_st = env_f.get("state_type"), model_f.get("state_type")
    if env_st and model_st and env_st != model_st:
        gaps.append(
            Gap(
                category,
                f"{label}: state_type mismatch",
                f"env.state_type={env_st} vs model.state_type={model_st}",
            )
        )

    env_repr, model_repr = env_f.get("state_representation"), model_f.get("state_representation")
    if env_repr and model_repr and env_repr != model_repr:
        gaps.append(
            Gap(
                category,
                f"{label}: state_representation mismatch",
                f"env.state_representation={env_repr} vs model.state_representation={model_repr}",
            )
        )

    env_frame, model_frame = env_f.get("frame"), model_f.get("frame")
    if env_frame and model_frame and env_frame != model_frame:
        gaps.append(
            Gap(
                category,
                f"{label}: frame mismatch",
                f"env.frame={env_frame} vs model.frame={model_frame}",
            )
        )

    env_shape, model_shape = env_f.get("shape"), model_f.get("shape")
    if env_shape != model_shape:
        gaps.append(
            Gap(
                category,
                f"{label}: shape mismatch",
                f"env.shape={env_shape} vs model.shape={model_shape}",
            )
        )

    env_units, model_units = env_f.get("units"), model_f.get("units")
    # Only flag units when the model declares concrete units. model.units="none" means
    # dimensionless/normalized on the model side — env may still carry physical units (m, rad)
    # without implying a mismatch (avoids noisy false positives e.g. gripper qpos in meters).
    if (
        env_units
        and model_units
        and model_units != "none"
        and env_units != model_units
    ):
        gaps.append(
            Gap(
                category,
                f"{label}: units mismatch",
                f"env.units={env_units} vs model.units={model_units}",
            )
        )

    # Model-side normalization is expected per SPEC (§6.2) — not reported as a gap here;
    # the adapter always applies the model's processor/denorm using env raw values + stats.

    return gaps


def integration_review(
    env: dict,
    model: dict,
    *,
    supported: bool | None = None,
) -> IntegrationReview | None:
    """Analyze integration gaps for a robot_type match. Returns None if no match."""
    robot_type = env.get("robot_type", "?")
    if supported is None:
        supported = match(model, robot_type)
    if not supported:
        return None

    obs_pairs = pair_observations(env, model)
    action = match_actions(env, model)

    env_images = sum(1 for (_, ef), _ in obs_pairs if ef and _is_image(ef))
    env_vectors = sum(1 for (_, ef), _ in obs_pairs if ef and not _is_image(ef))

    scope = [
        f"robot_type={robot_type!r} (gate)",
        f"obs: {env_images} image(s) + {env_vectors} vector(s), positional pairing",
    ]
    if action.matched:
        chunk = model.get("chunk_size")
        chunk_note = f", chunk_size={chunk}" if chunk else ""
        scope.append(f"act: [{action.signature}]{chunk_note}")
    else:
        scope.append(f"act: NO match for [{action.signature}]")

    problems: list[Gap] = []

    for (env_name, env_f), (model_name, model_f) in obs_pairs:
        problems.extend(_compare_feature_pair(env_name, env_f, model_name, model_f, category="obs"))

    if action.matched:
        for (env_name, env_f), (model_name, model_f) in action.pairs:
            problems.extend(
                _compare_feature_pair(env_name, env_f, model_name, model_f, category="act")
            )
    else:
        problems.append(
            Gap(
                "act",
                "action signature mismatch",
                f"env signature={action.signature}, "
                f"model signature={action.model_signature}",
            )
        )

    env_rate, model_rate = env.get("control_rate"), model.get("control_rate")
    if env_rate and model_rate and env_rate != model_rate:
        problems.append(
            Gap(
                "global",
                "control_rate mismatch",
                f"env.control_rate={env_rate} vs model.control_rate={model_rate}",
            )
        )

    return IntegrationReview(scope=scope, problems=problems)
