"""Typed request/result contracts for the HUD training (RL) service."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# A built-in, server-side loss name: an open string, since the valid set is
# provider-defined (see TrainingClient.available_losses).
LossFn = str


class BuiltinLoss(StrEnum):
    """Common Tinker-backed loss names; ``TrainingClient.available_losses`` is authoritative."""

    CROSS_ENTROPY = "cross_entropy"  # supervised; imitate sampled tokens
    IMPORTANCE_SAMPLING = "importance_sampling"  # on-policy PG, rollout-logprob ratio
    PPO = "ppo"  # clipped-surrogate PG
    CISPO = "cispo"  # clipped IS policy optimization
    DRO = "dro"  # direct reward optimization


class TrajectorySample(BaseModel):
    """One turn's token-level training data (mirrors ``hud.agents`` ``Sample``).

    ``output_logprobs`` are per-output-token logprobs under the sampling policy.
    """

    model_config = ConfigDict(extra="forbid")

    # Text-only prompts; multimodal prompts carry prompt_chunks instead.
    prompt_token_ids: list[int] | None = None
    # Serialized ModelInput chunks (text and images); authoritative when present.
    prompt_chunks: list[dict[str, Any]] | None = None
    output_token_ids: list[int]
    output_logprobs: list[float] = Field(default_factory=list[float])


class TrajectoryPayload(BaseModel):
    """An inline trajectory submitted for training (alternative to a ``trace_id``).

    ``trace_id`` is optional provenance when the trajectory also exists server-side.
    """

    model_config = ConfigDict(extra="forbid")

    samples: list[TrajectorySample] = Field(min_length=1)
    reward: float
    trace_id: str | None = None


# A single training input: a recorded trajectory by id, or an inline trajectory.
TrainInput = str | TrajectoryPayload


class ForwardBackwardRequest(BaseModel):
    """Accumulate gradients for one batch of trajectories on the model session."""

    model_config = ConfigDict(extra="forbid")

    inputs: list[TrainInput] = Field(min_length=1)
    loss_fn: LossFn = "importance_sampling"
    # Forwarded verbatim to the provider's loss; keys are provider-defined.
    loss_fn_config: dict[str, float] | None = None
    # GRPO advantages normalize within contiguous groups of this size; None is one group.
    group_size: int | None = Field(default=None, ge=1)
    reward_scale: float = 1.0
    num_substeps: int = Field(default=1, ge=1)


class ForwardBackwardResult(BaseModel):
    """Outcome of a ``forward_backward`` call (gradients accumulated, not applied)."""

    model_config = ConfigDict(extra="forbid")

    metrics: dict[str, float]
    num_datums: int


class OptimStepRequest(BaseModel):
    """Apply the accumulated gradients and checkpoint the new weights."""

    model_config = ConfigDict(extra="forbid")

    learning_rate: float = Field(gt=0)
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    # Adam weight decay, as in torch AdamW.
    weight_decay: float = Field(default=0.0, ge=0)


class OptimStepResult(BaseModel):
    """Outcome of an ``optim_step`` call after checkpointing and promotion."""

    model_config = ConfigDict(extra="forbid")

    step: int
    checkpoint_id: str
    sampler_path: str
    state_path: str
    # Gateway model string now serving the promoted weights (typically unchanged
    # across steps; the active checkpoint behind it advances).
    model: str


class CheckpointResponse(BaseModel):
    """One node in a model's checkpoint tree; each ``optim_step`` appends one.

    ``is_active`` marks the head the gateway serves; ``prev_model_checkpoint_id`` links
    to its parent.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    name: str | None = None
    checkpoint_name: str | None = None
    prev_model_checkpoint_id: str | None = None
    is_active: bool = False
    created_at: str | None = None
    num_traces: int | None = None
    num_datums: int | None = None
    num_tokens: int | None = None
    mean_reward: float | None = None
    learning_rate: float | None = None
    loss_fn: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)


class ForwardRequest(BaseModel):
    """Run a current-policy forward pass over a batch of trajectories."""

    model_config = ConfigDict(extra="forbid")

    inputs: list[TrainInput] = Field(min_length=1)
    # Contiguous groups for caller-side advantage normalization; None is one group.
    group_size: int | None = Field(default=None, ge=1)
    reward_scale: float = 1.0


class TrainingDatum(BaseModel):
    """Per-datum fields a custom loss reads.

    Datums from one trajectory share ``traj_idx``; ``group_idx`` is set only when
    ``group_size`` was given.
    """

    model_config = ConfigDict(extra="forbid")

    reward: float
    traj_idx: int
    group_idx: int | None = None


class DatumTensors(TrainingDatum):
    """LLM per-datum token tensors of equal length.

    ``logprobs`` are under the current policy, ``sampling_logprobs`` under the rollout
    policy, and ``mask`` is 1.0 on action tokens and 0.0 on observation tokens.
    """

    logprobs: list[float]
    sampling_logprobs: list[float]
    mask: list[float]


class ForwardResult(BaseModel):
    """A forward-pass handle plus the per-datum tensors to compute a loss over."""

    model_config = ConfigDict(extra="forbid")

    forward_id: str
    data: list[DatumTensors]


class BackwardRequest(BaseModel):
    """Apply caller-computed per-token gradients against a prior forward pass."""

    model_config = ConfigDict(extra="forbid")

    forward_id: str
    # weights[d][t] = -dC/dlogprobs for datum d, token t. Aligned with the
    # ``ForwardResult.data[d].logprobs`` returned by the matching forward.
    weights: list[list[float]]
    metrics: dict[str, float] = Field(default_factory=dict)
