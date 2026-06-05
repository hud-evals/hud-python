"""Remote rollout submission (v6) — submit a Taskset's variants to HUD infra.

Mirrors the legacy ``hud.datasets.utils.submit_rollouts`` shape, but over the new
:class:`~hud.eval.variant.Variant` (serialized to a portable env-ref + task + args).
The backend contract for running v6 variants remotely is **not finalized**, so the
endpoint call is left as a seam — wire it once the platform accepts variant
payloads.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .variant import Variant

logger = logging.getLogger("hud.eval.remote")

# Mirror of the legacy batch endpoint; confirm/replace when the v6 backend lands.
_RUN_LIST_PATH = "/v1/rollouts/run_list"


def _build_requests(
    variants: list[Variant],
    *,
    job_id: str,
    agent: dict[str, Any],
    group: int,
) -> list[dict[str, Any]]:
    """One request per variant x group; each carries the serialized env-ref + agent spec."""
    requests: list[dict[str, Any]] = []
    for variant in variants:
        spec = variant.to_dict()  # {"env": <ref>, "task": ..., "args": {...}}
        group_id = (job_id + ":" + spec["task"]) if group > 1 else None
        requests.extend(
            {**spec, "job_id": job_id, "group_id": group_id, "agent": agent}
            for _ in range(group)
        )
    return requests


async def submit_rollouts(
    variants: list[Variant],
    *,
    job_id: str,
    agent: dict[str, Any],
    group: int = 1,
    batch_size: int = 50,
) -> list[str]:
    """Submit variant rollouts to HUD for remote execution; return trace ids.

    TODO: the v6 remote-execution backend contract isn't defined yet. This builds
    the batched payload (mirroring the legacy ``/v1/rollouts/run_list`` flow) but
    the submission is intentionally unwired — implement once the platform accepts
    variant payloads.
    """
    from hud.settings import settings

    if not settings.api_key:
        raise ValueError("HUD_API_KEY is required for remote execution")

    requests = _build_requests(variants, job_id=job_id, agent=agent, group=group)
    logger.info("prepared %d remote rollout request(s) for job %s", len(requests), job_id)

    raise NotImplementedError(
        "v6 remote rollout submission is not wired yet: POST the batched payload to "
        f"{settings.hud_api_url.rstrip('/')}{_RUN_LIST_PATH} once the backend accepts "
        "variant (env-ref + task + args) payloads. The request builder is ready.",
    )


__all__ = ["submit_rollouts"]
