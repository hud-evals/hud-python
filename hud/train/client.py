"""Client for the HUD training (RL) service."""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import UUID

from pydantic import BaseModel

from hud.agents.types import AgentStep
from hud.settings import settings
from hud.train.types import (
    BackwardRequest,
    CheckpointResponse,
    DatumTensors,
    ForwardBackwardRequest,
    ForwardBackwardResult,
    ForwardRequest,
    ForwardResult,
    LossFn,
    OptimStepRequest,
    OptimStepResult,
    TrainInput,
    TrajectoryPayload,
)
from hud.utils.exceptions import HudAuthenticationError
from hud.utils.gateway import resolve_gateway_model
from hud.utils.platform import PlatformClient
from hud.utils.requests import make_request

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import torch

    from hud.types import Trace

    # A custom loss over a forward pass: given the per-datum tensors and the
    # current-policy logprobs as differentiable leaves, return (loss, metrics).
    CustomLossFn = Callable[
        [list[DatumTensors], list["torch.Tensor"]],
        tuple["torch.Tensor", dict[str, float]],
    ]

T = TypeVar("T")


class _Losses(BaseModel):
    losses: list[str]


logger = logging.getLogger("hud.train.client")


def _trace_to_input(trace: Trace) -> TrainInput:
    if trace.stop_reason == "timeout":
        raise ValueError("timed-out runs require an explicit training reward")
    if trace.grade.is_error:
        raise ValueError(
            f"trace {trace.trace_id} has no valid grade, so its reward is not a training signal; "
            "leave out its GRPO group or pass a TrajectoryPayload with an explicit reward"
        )
    samples = trace.collect(lambda step: step.sample if isinstance(step, AgentStep) else None)
    turns = [sample for sample in samples if sample.output_token_ids]
    if turns:
        return TrajectoryPayload(samples=turns, reward=trace.reward, trace_id=trace.trace_id)
    if trace.trace_id is not None:
        return trace.trace_id
    raise ValueError(
        "trace carries neither token-level samples nor a trace_id to train on; "
        "it must come from a trainable-model rollout (local) or a reported trace (remote)"
    )


def _to_inputs(trajectories: Sequence[str | Trace | TrajectoryPayload]) -> list[TrainInput]:
    return [
        item if isinstance(item, str | TrajectoryPayload) else _trace_to_input(item)
        for item in trajectories
    ]


def _check_groups(
    trajectories: Sequence[str | Trace | TrajectoryPayload],
    group_size: int | None,
) -> None:
    """Raise on incomplete or interleaved GRPO groups; warn when none has reward spread.

    The service groups contiguous slices of ``group_size`` (the whole batch when ``None``),
    so when every item is a trace carrying a ``group_id``, each slice must be one of those
    groups. No spread zeroes every advantage, a legitimate outcome for one batch.
    """
    n = len(trajectories)
    if group_size is not None and n % group_size != 0:
        raise ValueError(
            f"{n} trajectories do not divide evenly into groups of {group_size}; "
            "GRPO normalizes advantages within each group, so every group must be full"
        )
    run_group_ids = [
        item.group_id for item in trajectories if not isinstance(item, str | TrajectoryPayload)
    ]
    keyed = len(run_group_ids) == n and all(g is not None for g in run_group_ids)
    if keyed and group_size is not None:
        counts = Counter(run_group_ids)
        if incomplete := [group_id for group_id, count in counts.items() if count != group_size]:
            raise ValueError(f"incomplete GRPO groups: {incomplete}")
        if any(
            len(set(run_group_ids[start : start + group_size])) != 1
            for start in range(0, n, group_size)
        ):
            raise ValueError(
                f"GRPO groups must be contiguous: the service groups every {group_size} "
                "consecutive trajectories, so keep runs with the same group_id together"
            )

    inline = [item for item in trajectories if not isinstance(item, str)]
    if len(inline) != n or n < 2:
        return
    size = group_size or n
    reward_groups = [
        [item.reward for item in inline[start : start + size]] for start in range(0, n, size)
    ]
    if all(len(set(group)) == 1 for group in reward_groups):
        logger.warning(
            "no GRPO group has any reward spread, so every advantage normalizes to "
            "zero and this pass accumulates no gradient. Vary task difficulty "
            "until rollouts within a group disagree."
        )


class TrainingClient:
    """Train a model through the HUD training service.

    One handle on a model (a gateway slug or id) whose weights training advances in
    place. As in Tinker, ``forward_backward`` accumulates gradients and ``optim_step``
    applies them; ``step`` chains the two.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        api_url: str | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key or settings.api_key
        self._rl_url = (base_url or settings.hud_rl_url).rstrip("/")
        self._api_url = (api_url or settings.hud_api_url).rstrip("/")
        self._model_id: str | None = None

    async def _resolve_model_id(self) -> str:
        """The id the service keys on: a uuid as-is, else the catalog entry's, cached."""
        if self._model_id is None:
            try:
                self._model_id = str(UUID(self.model))
            except ValueError:
                if not self._api_key:
                    raise HudAuthenticationError(
                        "HUD_API_KEY is required to resolve a model slug"
                    ) from None
                platform = PlatformClient(self._api_url, self._api_key)
                entry = await asyncio.to_thread(resolve_gateway_model, self.model, platform)
                self._model_id = entry.id
        return self._model_id

    async def _train_url(self, suffix: str) -> str:
        model_id = await self._resolve_model_id()
        return f"{self._rl_url}/v1/models/{model_id}/train/{suffix}"

    async def _post(self, suffix: str, payload: dict[str, Any], reply: type[T]) -> T:
        url = await self._train_url(suffix)
        # Never auto-retry: gradients accumulate in place (a retry double-counts a batch)
        # and optim_step checkpoints (a retry can save a step twice). The server owns
        # recovery.
        return await make_request(
            "POST", url, json=payload, api_key=self._api_key, max_retries=0, reply=reply
        )

    async def _get(self, suffix: str, reply: type[T]) -> T:
        url = await self._train_url(suffix)
        return await make_request("GET", url, api_key=self._api_key, reply=reply)

    async def available_losses(self) -> list[str]:
        """The built-in ``loss_fn`` names this model's provider supports.

        ``BuiltinLoss`` lists common ones; this list is authoritative.
        """
        return (await self._get("losses", _Losses)).losses

    async def optim_step(
        self,
        *,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> OptimStepResult:
        """Apply accumulated gradients and checkpoint; the gateway then serves the new weights.

        Never retried automatically, since a duplicate submit would save a step twice.
        """
        request = OptimStepRequest(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
        )
        return await self._post("optim-step", request.model_dump(), OptimStepResult)

    async def checkpoints(self) -> list[CheckpointResponse]:
        """The model's checkpoint tree, oldest first, with each node's rewards and metrics."""
        model_id = await self._resolve_model_id()
        # The checkpoints endpoint lives on the catalog API.
        url = f"{self._api_url}/v2/models/{model_id}/checkpoints"
        return await make_request("GET", url, api_key=self._api_key, reply=list[CheckpointResponse])

    async def head(self) -> CheckpointResponse | None:
        """The checkpoint the gateway serves, or ``None`` while it serves the base weights."""
        return next((c for c in await self.checkpoints() if c.is_active), None)

    async def set_head(self, checkpoint_id: str) -> None:
        """Promote a checkpoint to head; the next ``optim_step`` extends the tree from there."""
        model_id = await self._resolve_model_id()
        url = f"{self._api_url}/v2/models/{model_id}/head"
        await make_request("PUT", url, json={"checkpoint_id": checkpoint_id}, api_key=self._api_key)

    async def forward_backward(
        self,
        trajectories: Sequence[str | Trace | TrajectoryPayload],
        *,
        loss_fn: LossFn = "importance_sampling",
        loss_fn_config: dict[str, float] | None = None,
        group_size: int | None = None,
        reward_scale: float = 1.0,
        num_substeps: int = 1,
    ) -> ForwardBackwardResult:
        """Accumulate gradients for a batch of trajectories with a built-in loss.

        Each trajectory is a ``trace_id`` (resolved server-side), a graded trace, or a
        hand-built ``TrajectoryPayload``. Advantages normalize within contiguous groups
        of ``group_size``, or across the batch when ``None``. ``loss_fn_config`` keys
        are provider-defined.
        """
        inputs = _to_inputs(trajectories)
        _check_groups(trajectories, group_size)
        request = ForwardBackwardRequest(
            inputs=inputs,
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config,
            group_size=group_size,
            reward_scale=reward_scale,
            num_substeps=num_substeps,
        )
        return await self._post("forward-backward", request.model_dump(), ForwardBackwardResult)

    async def forward(
        self,
        trajectories: Sequence[str | Trace | TrajectoryPayload],
        *,
        group_size: int | None = None,
        reward_scale: float = 1.0,
    ) -> ForwardResult:
        """Run a current-policy forward pass and return per-token tensors.

        The first half of a custom loss, finished by ``backward``;
        ``forward_backward_custom`` runs both.
        """
        inputs = _to_inputs(trajectories)
        _check_groups(trajectories, group_size)
        request = ForwardRequest(
            inputs=inputs,
            group_size=group_size,
            reward_scale=reward_scale,
        )
        return await self._post("forward", request.model_dump(), ForwardResult)

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
        ``ForwardResult.data`` from the matching ``forward``.
        """
        request = BackwardRequest(forward_id=forward_id, weights=weights, metrics=metrics or {})
        return await self._post("backward", request.model_dump(), ForwardBackwardResult)

    async def forward_backward_custom(
        self,
        trajectories: Sequence[str | Trace | TrajectoryPayload],
        loss_fn: CustomLossFn,
        *,
        group_size: int | None = None,
        reward_scale: float = 1.0,
    ) -> ForwardBackwardResult:
        """Accumulate gradients with a caller-authored, client-side loss.

        ``loss_fn`` runs under torch autograd over the forward pass's ``DatumTensors``
        and current-policy logprobs. Requires the ``train`` extra.
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
        trajectories: Sequence[str | Trace | TrajectoryPayload],
        *,
        learning_rate: float,
        loss_fn: LossFn = "importance_sampling",
        loss_fn_config: dict[str, float] | None = None,
        group_size: int | None = None,
        reward_scale: float = 1.0,
        num_substeps: int = 1,
        weight_decay: float = 0.0,
    ) -> OptimStepResult:
        """Run one ``forward_backward`` then one ``optim_step``.

        The ``forward_backward`` metrics are discarded.
        """
        await self.forward_backward(
            trajectories,
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config,
            group_size=group_size,
            reward_scale=reward_scale,
            num_substeps=num_substeps,
        )
        return await self.optim_step(learning_rate=learning_rate, weight_decay=weight_decay)
