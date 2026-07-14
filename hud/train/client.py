"""Client for the HUD training (RL) service.

A thin, async HTTP wrapper over the model-id-keyed training endpoints. One client
instance targets one model string (the same string used for inference through the
HUD gateway); training advances the weights behind that string in place.

Every training call takes a sequence of trajectories, each either a ``trace_id``
string (the service resolves recorded tokens + reward) or a :class:`hud.Run`
(its trajectory + reward are extracted and sent inline). The two can be mixed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hud.agents.types import AgentStep
from hud.train.base import BaseTrainingClient
from hud.train.types import (
    BackwardRequest,
    DatumTensors,
    ForwardBackwardRequest,
    ForwardBackwardResult,
    ForwardRequest,
    ForwardResult,
    LossFn,
    OptimStepResult,
    TrainInput,
    TrajectoryPayload,
    TrajectorySample,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import torch

    from hud.eval.run import Run

    # A custom loss over a forward pass: given the per-datum tensors and the
    # current-policy logprobs as differentiable leaves, return (loss, metrics).
    CustomLossFn = Callable[
        [list[DatumTensors], list["torch.Tensor"]],
        tuple["torch.Tensor", dict[str, float]],
    ]


def _run_to_input(run: Run) -> TrainInput:
    """Turn a graded :class:`hud.Run` into a training input: inline payload when
    the run carries token-level samples (local rollout), else its ``trace_id``
    (remote rollout, resolved server-side)."""
    samples = run.trace.collect(lambda step: step.sample if isinstance(step, AgentStep) else None)
    turns = [
        TrajectorySample(
            prompt_token_ids=sample.prompt_token_ids or None,
            prompt_chunks=sample.prompt_chunks,
            output_token_ids=sample.output_token_ids,
            output_logprobs=sample.output_logprobs,
        )
        for sample in samples
        if sample.output_token_ids
    ]
    if turns:
        return TrajectoryPayload(samples=turns, reward=run.reward, trace_id=run.trace_id)
    if run.trace_id is not None:
        return run.trace_id
    raise ValueError(
        "run carries neither token-level samples nor a trace_id to train on; "
        "it must come from a trainable-model rollout (local) or a reported trace (remote)"
    )


def _to_inputs(trajectories: Sequence[str | Run | TrajectoryPayload]) -> list[TrainInput]:
    """Normalize a mix of ``trace_id`` strings, inline ``TrajectoryPayload``s, and
    ``Run``s to wire inputs. The first two pass through; a ``Run`` is converted."""
    return [
        item if isinstance(item, str | TrajectoryPayload) else _run_to_input(item)
        for item in trajectories
    ]


def _check_groups(n: int, group_size: int | None) -> None:
    """Fail before a forward pass if the batch can't form full GRPO groups: an
    incomplete final group gets a skewed advantage baseline. Cheap, client-side,
    no round-trip — unlike per-group reward spread, which only the service sees."""
    if group_size is not None and n % group_size != 0:
        raise ValueError(
            f"{n} trajectories do not divide evenly into groups of {group_size}; "
            "GRPO normalizes advantages within each group, so every group must be full"
        )


class TrainingClient(BaseTrainingClient):
    """Train an LLM model through the HUD training service.

    The LLM modality client. Mirrors the Tinker split between gradient
    accumulation (:meth:`forward_backward`) and the optimizer update
    (:meth:`optim_step`, inherited from :class:`BaseTrainingClient`). :meth:`step`
    chains both for the common case; :meth:`forward_backward_custom` runs a
    caller-authored loss over per-token logprobs (:class:`DatumTensors`).
    """

    async def forward_backward(
        self,
        trajectories: Sequence[str | Run | TrajectoryPayload],
        *,
        loss_fn: LossFn = "importance_sampling",
        loss_fn_config: dict[str, float] | None = None,
        group_size: int | None = None,
        reward_scale: float = 1.0,
        num_substeps: int = 1,
    ) -> ForwardBackwardResult:
        """Accumulate gradients for a batch of trajectories with a built-in loss.

        Each trajectory is a ``trace_id`` (resolved server-side), a ``Run`` (its
        tokens + reward sent inline), or an inline :class:`TrajectoryPayload`
        (hand-built tokens + reward — e.g. the opponent side of a self-play game).
        Advantages are normalized within contiguous groups of ``group_size`` (all
        trajectories as one group when ``None``). ``loss_fn_config`` forwards
        provider-specific hyperparameters to the loss; ``None`` uses provider
        defaults (the supported keys are provider-defined, so prefer defaults).
        """
        inputs = _to_inputs(trajectories)
        _check_groups(len(inputs), group_size)
        request = ForwardBackwardRequest(
            inputs=inputs,
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config,
            group_size=group_size,
            reward_scale=reward_scale,
            num_substeps=num_substeps,
        )
        # exclude_none so optional fields a (possibly older) server doesn't know
        # yet — e.g. a null prompt_chunks/trace_id — aren't sent and rejected.
        data = await self._post("forward-backward", request.model_dump(exclude_none=True))
        return ForwardBackwardResult.model_validate(data)

    async def forward(
        self,
        trajectories: Sequence[str | Run | TrajectoryPayload],
        *,
        group_size: int | None = None,
        reward_scale: float = 1.0,
    ) -> ForwardResult:
        """Run a current-policy forward pass and return per-token tensors.

        The first half of the custom-loss path: compute a loss over the returned
        :class:`DatumTensors` (current-policy logprobs as differentiable leaves),
        then send the gradients through :meth:`backward`. For the common case use
        :meth:`forward_backward_custom`, which wires both halves together.
        """
        inputs = _to_inputs(trajectories)
        _check_groups(len(inputs), group_size)
        request = ForwardRequest(
            inputs=inputs,
            group_size=group_size,
            reward_scale=reward_scale,
        )
        data = await self._post("forward", request.model_dump(exclude_none=True))
        return ForwardResult.model_validate(data)

    async def backward(
        self,
        forward_id: str,
        weights: list[list[float]],
        *,
        metrics: dict[str, float] | None = None,
    ) -> ForwardBackwardResult:
        """Accumulate gradients from a caller-computed loss against a forward pass.

        ``weights[d][t]`` is ``-dC/dlogprobs`` for datum ``d``, token ``t`` (the
        Tinker cross-entropy backward convention), aligned with the
        :attr:`ForwardResult.data` from the matching :meth:`forward`.
        """
        request = BackwardRequest(forward_id=forward_id, weights=weights, metrics=metrics or {})
        data = await self._post("backward", request.model_dump())
        return ForwardBackwardResult.model_validate(data)

    async def forward_backward_custom(
        self,
        trajectories: Sequence[str | Run | TrajectoryPayload],
        loss_fn: CustomLossFn,
        *,
        group_size: int | None = None,
        reward_scale: float = 1.0,
    ) -> ForwardBackwardResult:
        """Accumulate gradients with a caller-authored, client-side loss.

        ``forward`` → run ``loss_fn`` locally (torch autograd) → ship per-token
        gradients to ``backward``. Any differentiable loss over π_θ and the
        :class:`DatumTensors` scalars works (e.g. GLM double-sided IS). Requires
        torch (``pip install 'hud[train]'``).
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "forward_backward_custom requires torch; install 'hud[train]'"
            ) from exc

        forward = await self.forward(trajectories, group_size=group_size, reward_scale=reward_scale)
        logprob_leaves = [
            torch.tensor(datum.logprobs, dtype=torch.float32).requires_grad_(True)
            for datum in forward.data
        ]
        loss, metrics = loss_fn(forward.data, logprob_leaves)
        loss.backward()

        weights: list[list[float]] = []
        for leaf in logprob_leaves:
            if leaf.grad is None:
                raise ValueError("custom loss produced no gradient for a datum's logprobs")
            weights.append((-leaf.grad).detach().tolist())

        return await self.backward(forward.forward_id, weights, metrics=metrics)

    async def step(
        self,
        trajectories: Sequence[str | Run | TrajectoryPayload],
        *,
        learning_rate: float,
        loss_fn: LossFn = "importance_sampling",
        loss_fn_config: dict[str, float] | None = None,
        group_size: int | None = None,
        reward_scale: float = 1.0,
        num_substeps: int = 1,
        weight_decay: float = 0.0,
    ) -> OptimStepResult:
        """Convenience: one ``forward_backward`` followed by one ``optim_step``."""
        await self.forward_backward(
            trajectories,
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config,
            group_size=group_size,
            reward_scale=reward_scale,
            num_substeps=num_substeps,
        )
        return await self.optim_step(learning_rate=learning_rate, weight_decay=weight_decay)
