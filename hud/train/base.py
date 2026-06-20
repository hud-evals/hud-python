"""Shared training lifecycle: the model handle, HTTP plumbing, and the
modality-independent ``optim_step``. Modality clients (e.g.
:class:`hud.train.TrainingClient`) subclass and add ``forward_backward`` etc.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote
from uuid import UUID

from hud.settings import settings
from hud.train.types import CheckpointResponse, OptimStepRequest, OptimStepResult
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
        directly, a slug/name is looked up once via the catalog and cached.

        Resolution order:
        1. UUID → used directly.
        2. Slug (model_name) → GET /v2/models/resolve (fast path).
        3. Display name or unknown string → fall back to list_gateway_models(),
           which matches id | name | model_name — same logic as create_agent().
        """
        if self._model_id is not None:
            return self._model_id
        try:
            self._model_id = str(UUID(self.model))
        except ValueError:
            url = f"{self._api_url}/v2/models/resolve?model={quote(self.model, safe='')}"
            try:
                data = await make_request("GET", url, api_key=self._api_key)
                self._model_id = str(data["id"])
            except Exception as exc:
                # /v2/models/resolve only matches model_name (slug), not the
                # display name. Fall back to the full model list, which matches
                # id | name | model_name — consistent with create_agent().
                from hud.utils.gateway import list_gateway_models

                for gm in list_gateway_models():
                    if self.model in (gm.id, gm.name, gm.model_name):
                        resolved = gm.id or gm.model_name
                        if resolved:
                            self._model_id = resolved
                            break
                else:
                    raise ValueError(
                        f"Model {self.model!r} not found. "
                        "Run `hud models` to list available models."
                    ) from exc
        return self._model_id

    async def _train_url(self, suffix: str) -> str:
        model_id = await self._resolve_model_id()
        return f"{self._base_url}/v1/models/{model_id}/train/{suffix}"

    async def _post(
        self, suffix: str, payload: dict[str, Any], *, max_retries: int = 0
    ) -> dict[str, Any]:
        url = await self._train_url(suffix)
        # Training POSTs are not safe to auto-retry: forward_backward/backward
        # accumulate gradients in place (a retry double-counts a batch) and
        # optim_step saves a checkpoint (a retry can save a step twice and poison
        # the run). All callers pass max_retries=0; the server owns recovery.
        return await make_request(
            "POST", url, json=payload, api_key=self._api_key, max_retries=max_retries
        )

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
        checkpoint, so the gateway serves the updated weights.

        One-shot, never retried: a duplicate submit is what poisons a training
        run (the same step number saved twice). The server owns recovery — it
        adopts an already-saved step on a colliding save and rebuilds a poisoned
        run from head on the next call — so a failed step is safe to simply
        re-run from the training loop."""
        request = OptimStepRequest(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
        )
        data = await self._post("optim-step", request.model_dump(), max_retries=0)
        return OptimStepResult.model_validate(data)

    async def checkpoints(self) -> list[CheckpointResponse]:
        """The model's checkpoint tree (oldest first), each node carrying its
        rewards, loss, token/datum counts, and a ``metrics`` blob. The
        programmatic twin of ``hud models checkpoints`` — poll it inside a
        training loop to watch ``mean_reward``, KL, and clip fraction live."""
        model_id = await self._resolve_model_id()
        # The checkpoints endpoint lives on the catalog API and returns a JSON
        # array (make_request is typed for the common object response).
        url = f"{self._api_url}/v2/models/{model_id}/checkpoints"
        data: Any = await make_request("GET", url, api_key=self._api_key)
        return [CheckpointResponse.model_validate(c) for c in data]

    async def head(self) -> CheckpointResponse | None:
        """The active checkpoint (the weights the gateway serves now), or
        ``None`` when the model still serves its base weights."""
        return next((c for c in await self.checkpoints() if c.is_active), None)

    async def set_head(self, checkpoint_id: str) -> None:
        """Promote a checkpoint to head — a rollback to, or branch from, that
        node; the next :meth:`optim_step` extends the tree from there.

        Use this when the reward function or environment changes underneath a
        run: continuing from a stale head fine-tunes a policy shaped by the old
        objective. Roll back to a clean checkpoint (or fork a fresh model) so the
        run measures the new objective, not a contaminated starting point."""
        model_id = await self._resolve_model_id()
        url = f"{self._api_url}/v2/models/{model_id}/head"
        await make_request("PUT", url, json={"checkpoint_id": checkpoint_id}, api_key=self._api_key)
