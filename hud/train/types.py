"""Typed request/result contracts for the HUD training (RL) service.

Each training input is a trajectory: a ``trace_id`` (resolved server-side) or an
inline :class:`TrajectoryPayload`; the forms mix and order is preserved. Custom
client-side losses use :class:`ForwardResult` / :class:`BackwardRequest`.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

# A built-in, server-side loss name. Open string, not an enum: the valid set is
# provider-defined (discover it via ``TrainingClient.available_losses()``). For a
# loss a provider lacks, use ``TrainingClient.forward_backward_custom``.
LossFn = str


class BuiltinLoss(StrEnum):
    """Common Tinker-backed loss names (each *is* a ``str``; not authoritative —
    see :meth:`TrainingClient.available_losses`)."""

    CROSS_ENTROPY = "cross_entropy"  # supervised; imitate sampled tokens
    IMPORTANCE_SAMPLING = "importance_sampling"  # on-policy PG, rollout-logprob ratio
    PPO = "ppo"  # clipped-surrogate PG
    CISPO = "cispo"  # clipped IS policy optimization
    DRO = "dro"  # direct reward optimization


class TrajectorySample(BaseModel):
    """One turn's token-level training data (mirrors :class:`hud.agents` ``Sample``).

    ``output_logprobs`` are the per-output-token logprobs under the *sampling*
    policy q (the behavior proxy used by importance sampling).
    """

    model_config = ConfigDict(extra="forbid")

    prompt_token_ids: list[int]
    output_token_ids: list[int]
    output_logprobs: list[float] = Field(default_factory=list[float])


class TrajectoryPayload(BaseModel):
    """An inline trajectory submitted for training (alternative to a ``trace_id``).

    Carries the ordered per-turn samples plus the trajectory reward. ``trace_id``
    is optional provenance when the trajectory also exists server-side.
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
    # Provider-specific loss hyperparameters forwarded verbatim to the provider's
    # loss (Tinker ``loss_fn_config``). The supported keys are provider-defined
    # and not all losses accept config; ``None`` uses the provider defaults.
    loss_fn_config: dict[str, float] | None = None
    # Trajectories are normalized for advantages within contiguous groups of this
    # size (GRPO). ``None`` treats all trajectories as a single group.
    group_size: int | None = Field(default=None, ge=1)
    reward_scale: float = 1.0
    num_substeps: int = Field(default=1, ge=1)


class ForwardBackwardResult(BaseModel):
    """Outcome of a ``forward_backward`` call (gradients accumulated, not applied)."""

    model_config = ConfigDict(extra="forbid")

    metrics: dict[str, float]
    num_datums: int


class OptimStepRequest(BaseModel):
    """Apply the accumulated gradients and checkpoint the new weights.

    This is a compound operation: optimizer step, save training state, save
    sampler weights, and advance the model's active sampler path so subsequent
    inference through the gateway serves the updated weights.
    """

    model_config = ConfigDict(extra="forbid")

    learning_rate: float = Field(gt=0)
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    # Adam weight decay (Tinker ``AdamParams``; matches torch AdamW). Default 0.
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
    """One node in a model's checkpoint tree (the programmatic form of a
    ``hud models checkpoints`` row).

    Each ``optim_step`` appends one. ``is_active`` marks the head the gateway
    serves; ``prev_model_checkpoint_id`` links to its parent. ``metrics`` is the
    provider's per-step blob (e.g. ``reward_std``/``reward_min``/``reward_max``,
    ``sampling_logprob_mean``, and ``tinker.*`` training stats). Extra fields the
    service returns (jobs, tasksets, reward groups) are ignored, not rejected.
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


# ── Custom-loss path ─────────────────────────────────────────────────────────
# Splits the server-side built-in loss so the caller authors it:
#   1. ``forward(inputs)`` returns per-token tensors (:class:`DatumTensors`).
#   2. caller computes a differentiable per-token loss over π_θ (torch autograd).
#   3. ``backward(forward_id, weights)`` applies ``weights = -dC/dlogprobs``.
# ``optim_step`` then applies + checkpoints as usual.


class ForwardRequest(BaseModel):
    """Run a current-policy forward pass over a batch of trajectories."""

    model_config = ConfigDict(extra="forbid")

    inputs: list[TrainInput] = Field(min_length=1)
    # Contiguous grouping for caller-side advantage normalization (e.g. GRPO).
    # ``None`` tags every trajectory into a single group.
    group_size: int | None = Field(default=None, ge=1)
    reward_scale: float = 1.0


class TrainingDatum(BaseModel):
    """Per-datum fields a custom loss reads: ``reward``, the source ``traj_idx``
    (datums from one trajectory share it), and ``group_idx`` (the GRPO group, set
    only when ``group_size`` was given; ``None`` otherwise)."""

    model_config = ConfigDict(extra="forbid")

    reward: float
    traj_idx: int
    group_idx: int | None = None


class DatumTensors(TrainingDatum):
    """LLM per-datum token tensors (aligned, equal length). ``logprobs`` are the
    current policy π_θ, ``sampling_logprobs`` the rollout policy q, and ``mask``
    is ``1.0`` on action tokens / ``0.0`` on observation tokens."""

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
