"""Shared training lifecycle: the model handle, HTTP plumbing, and the
modality-independent ``optim_step``. Modality clients (e.g.
:class:`hud.train.TrainingClient`) subclass and add ``forward_backward`` etc.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote
from uuid import UUID

from hud.settings import settings
from hud.train.types import OptimStepRequest, OptimStepResult
from hud.utils.requests import make_request


class BaseTrainingClient:
    """One model handle (a gateway slug or id) + the shared optimizer step.

    Training advances the weights behind the model string in place. The service
    keys on model id, so a slug is resolved once via the catalog and cached. Use
    a modality client such as :class:`hud.train.TrainingClient`, not this directly.
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
        """Resolve ``self.model`` to the id the service keys on: a uuid is used
        directly, a slug is looked up once via the catalog and cached."""
        if self._model_id is not None:
            return self._model_id
        try:
            self._model_id = str(UUID(self.model))
        except ValueError:
            url = f"{self._api_url}/v2/models/resolve?model={quote(self.model, safe='')}"
            data = await make_request("GET", url, api_key=self._api_key)
            self._model_id = str(data["id"])
        return self._model_id

    async def _train_url(self, suffix: str) -> str:
        model_id = await self._resolve_model_id()
        return f"{self._base_url}/v1/models/{model_id}/train/{suffix}"

    async def _post(self, suffix: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = await self._train_url(suffix)
        return await make_request("POST", url, json=payload, api_key=self._api_key)

    async def _get(self, suffix: str) -> dict[str, Any]:
        url = await self._train_url(suffix)
        return await make_request("GET", url, api_key=self._api_key)

    async def available_losses(self) -> list[str]:
        """The built-in ``loss_fn`` names this model's provider supports
        (authoritative; :class:`hud.train.BuiltinLoss` lists common ones)."""
        data = await self._get("losses")
        return list(data["losses"])

    async def optim_step(
        self,
        *,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> OptimStepResult:
        """Apply accumulated gradients, then checkpoint + promote: one compound
        step that saves state + sampler weights and advances the model's active
        checkpoint, so the gateway serves the updated weights."""
        request = OptimStepRequest(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
        )
        data = await self._post("optim-step", request.model_dump())
        return OptimStepResult.model_validate(data)
