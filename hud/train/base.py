"""The model handle and ``optim_step`` that modality clients such as ``TrainingClient`` share."""

from __future__ import annotations

import asyncio
from typing import Any, TypeVar
from uuid import UUID

from pydantic import BaseModel

from hud.settings import settings
from hud.train.types import CheckpointResponse, OptimStepRequest, OptimStepResult
from hud.utils.exceptions import HudAuthenticationError
from hud.utils.gateway import resolve_gateway_model
from hud.utils.platform import PlatformClient
from hud.utils.requests import make_request

T = TypeVar("T")


class _Losses(BaseModel):
    losses: list[str]


class BaseTrainingClient:
    """One model handle (a gateway slug or id) + the shared optimizer step.

    Training advances the weights behind the model string in place. Use a modality
    client such as ``TrainingClient``, not this directly.
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
        # RL training service (forward/backward/optim); catalog lives on the API.
        self._base_url = (base_url or settings.hud_rl_url).rstrip("/")
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
        return f"{self._base_url}/v1/models/{model_id}/train/{suffix}"

    async def _post(
        self, suffix: str, payload: dict[str, Any], reply: type[T], *, max_retries: int = 0
    ) -> T:
        url = await self._train_url(suffix)
        # Never auto-retry: gradients accumulate in place (a retry double-counts a batch)
        # and optim_step checkpoints (a retry can save a step twice). The server owns
        # recovery.
        return await make_request(
            "POST", url, json=payload, api_key=self._api_key, max_retries=max_retries, reply=reply
        )

    async def _get(self, suffix: str, reply: type[T]) -> T:
        url = await self._train_url(suffix)
        return await make_request("GET", url, api_key=self._api_key, reply=reply)

    async def available_losses(self) -> list[str]:
        """The built-in ``loss_fn`` names this model's provider supports
        (authoritative; ``hud.train.BuiltinLoss`` lists common ones)."""
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

        Never retried, since a duplicate submit saves a step twice; the server recovers,
        so a failed step is safe to re-run from the training loop.
        """
        request = OptimStepRequest(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
        )
        return await self._post("optim-step", request.model_dump(), OptimStepResult, max_retries=0)

    async def checkpoints(self) -> list[CheckpointResponse]:
        """The model's checkpoint tree, oldest first, with each node's rewards and metrics."""
        model_id = await self._resolve_model_id()
        # The checkpoints endpoint lives on the catalog API.
        url = f"{self._api_url}/v2/models/{model_id}/checkpoints"
        return await make_request("GET", url, api_key=self._api_key, reply=list[CheckpointResponse])

    async def head(self) -> CheckpointResponse | None:
        """The active checkpoint (the weights the gateway serves now), or
        ``None`` when the model still serves its base weights."""
        return next((c for c in await self.checkpoints() if c.is_active), None)

    async def set_head(self, checkpoint_id: str) -> None:
        """Promote a checkpoint to head; the next ``optim_step`` extends the tree from there."""
        model_id = await self._resolve_model_id()
        url = f"{self._api_url}/v2/models/{model_id}/head"
        await make_request("PUT", url, json={"checkpoint_id": checkpoint_id}, api_key=self._api_key)
