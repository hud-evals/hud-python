"""The robot step payloads a run's telemetry records: observations, inferences, video."""

from __future__ import annotations

from typing import Any, ClassVar, cast

from pydantic import BaseModel, ConfigDict, Field

from hud.types import ROBOT_STEP_SCHEMA, ImageContent, RobotStepSource, Step


class StateFeature(BaseModel):
    """An observation feature group's values and, when the contract has them, their labels."""

    model_config = ConfigDict(extra="forbid")

    names: list[str] = Field(default_factory=list[str])
    values: list[float] = Field(default_factory=list[float])


class ObservationStep(Step[RobotStepSource]):
    """What the policy saw at one control tick; ``tick`` pairs it with its ``InferenceStep``."""

    schema_tag: ClassVar[str] = ROBOT_STEP_SCHEMA
    source: RobotStepSource = "observation"

    tick: int = 0
    images: dict[str, ImageContent] = Field(default_factory=dict[str, ImageContent])
    state: dict[str, StateFeature] = Field(default_factory=dict[str, StateFeature])

    @classmethod
    def from_obs(
        cls,
        obs: dict[str, Any],
        *,
        tick: int = 0,
        obs_space: dict[str, Any] | None = None,
    ) -> ObservationStep:
        """Build a step's ``state`` from a raw ``robot`` obs, labelled by ``obs_space``."""
        obs_space = obs_space or {}
        state: dict[str, StateFeature] = {}
        for name, arr in obs.get("data", {}).items():
            if arr.ndim >= 2:
                continue  # camera frames travel as video, not per-tick images
            vec = arr.tolist()
            # Contract features keyed by this data key (as a dot segment) label the flat vector:
            # ordered slices that tile it become one group each (libero_pro), or one feature named
            # exactly like the key labels it whole (libero_ee_del). Otherwise it stays unlabelled.
            slices: list[tuple[int, int, str, list[str]]] = []
            direct: list[str] | None = None
            for feature_key, feature in obs_space.items():
                if name not in feature_key.split(".") or not isinstance(feature, dict):
                    continue
                feature_meta = cast("dict[str, Any]", feature)
                raw_names = feature_meta.get("names")
                labels = (
                    [str(n) for n in cast("list[Any]", raw_names)]
                    if isinstance(raw_names, list)
                    else []
                )
                order = feature_meta.get("order")
                if order is not None:
                    bounds = str(order).split("-")
                    slices.append(
                        (
                            int(bounds[0]),
                            int(bounds[-1]),
                            feature_key.split(".")[-1],
                            labels,
                        )
                    )
                elif feature_key.split(".")[-1] == name and len(labels) == len(vec):
                    direct = labels
            slices.sort()
            covered = [i for start, end, _, _ in slices for i in range(start, end + 1)]
            if covered == list(range(len(vec))):
                for start, end, key, labels in slices:
                    values = vec[start : end + 1]
                    state[key] = StateFeature(
                        names=labels if len(labels) == len(values) else [],
                        values=values,
                    )
            elif direct is not None:
                state[name] = StateFeature(names=direct, values=vec)
            else:
                state[name] = StateFeature(values=vec)
        return cls(tick=tick, state=state)


class InferenceStep(Step[RobotStepSource]):
    """The ``[T, A]`` action chunk the policy executed from one control tick.

    A single action is a length-1 chunk; ``tick`` matches the paired observation.
    """

    schema_tag: ClassVar[str] = ROBOT_STEP_SCHEMA
    source: RobotStepSource = "inference"

    tick: int = 0

    chunk: list[list[float]] = Field(default_factory=list[list[float]])
    chunk_length: int = 1
    # Per-action-dimension labels from the env contract, for the viewer's action plots.
    names: list[str] = Field(default_factory=list[str])


class VideoSegmentStep(Step[RobotStepSource]):
    """One CMAF (fragmented MP4) fragment of a camera's H.264 stream."""

    schema_tag: ClassVar[str] = ROBOT_STEP_SCHEMA
    source: RobotStepSource = "video_segment"

    camera: str = ""
    # Position in the camera's stream; ``index`` 0 is the init segment.
    index: int = 0
    fps: int = 10
    # A ``video`` content block (base64 MP4), which ingest offloads to S3 like an image.
    segment: dict[str, Any] = Field(default_factory=dict[str, Any])
