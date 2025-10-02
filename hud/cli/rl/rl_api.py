"""
Direct API functions for HUD RL remote endpoints using shared requests module.

This module provides functions for interacting with the HUD RL API server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from hud.settings import settings
from hud.shared.requests import make_request_sync

if TYPE_CHECKING:
    from collections.abc import Iterator


class RLModelInfo(BaseModel):
    """Model information from the API."""

    name: str
    base_model: str
    vllm_url: str | None = None
    trainer_name: str | None = None
    checkpoint_volume: str | None = None
    status: str = "pending"  # pending, deploying, ready, training, terminated
    created_at: str | None = None
    updated_at: str | None = None
    terminated_at: str | None = None


def create_model(name: str, base_model: str) -> dict[str, Any]:
    """Create a new model."""
    return make_request_sync(
        method="POST",
        url=f"{settings.hud_rl_url}/models",
        json={"name": name, "base_model": base_model},
        api_key=settings.api_key,
    )


def get_model(name: str) -> RLModelInfo:
    """Get model information."""
    response = make_request_sync(
        method="GET", url=f"{settings.hud_rl_url}/models/{name}", api_key=settings.api_key
    )
    return RLModelInfo(**response)


def list_models() -> list[RLModelInfo]:
    """List all models."""
    response = make_request_sync(
        method="GET", url=f"{settings.hud_rl_url}/models", api_key=settings.api_key
    )
    if not isinstance(response, list):
        response = [response]
    return [
        RLModelInfo(**(model if isinstance(model, dict) else model.__dict__)) for model in response
    ]


def deploy_vllm(model_name: str, gpu_type: str = "A100", gpu_count: int = 1) -> dict[str, Any]:
    """Deploy a vLLM server for a model."""
    return make_request_sync(
        method="POST",
        url=f"{settings.hud_rl_url}/models/{model_name}/deploy",
        json={"gpu_type": gpu_type, "gpu_count": gpu_count},
        api_key=settings.api_key,
    )


def stop_vllm(model_name: str) -> dict[str, Any]:
    """Stop the vLLM server for a model."""
    return make_request_sync(
        method="DELETE",
        url=f"{settings.hud_rl_url}/models/{model_name}/deploy",
        api_key=settings.api_key,
    )


def stop_training(model_name: str) -> dict[str, Any]:
    """Stop the training for a model."""
    return make_request_sync(
        method="DELETE",
        url=f"{settings.hud_rl_url}/models/{model_name}/training",
        api_key=settings.api_key,
    )


def launch_training(
    model_name: str,
    config: dict[str, Any],
    tasks: list[dict[str, Any]],
    gpu_type: str = "A100",
    gpu_count: int = 1,
) -> dict[str, Any]:
    """Launch a training run for a model."""
    return make_request_sync(
        method="POST",
        url=f"{settings.hud_rl_url}/models/{model_name}/training/launch",
        json={"config": config, "tasks": tasks, "gpu_type": gpu_type, "gpu_count": gpu_count},
        api_key=settings.api_key,
    )


def get_training_status(model_name: str) -> dict[str, Any]:
    """Get the status of a training run."""
    return make_request_sync(
        method="GET",
        url=f"{settings.hud_rl_url}/models/{model_name}/training/status",
        api_key=settings.api_key,
    )


def get_training_logs(model_name: str, lines: int = 100, follow: bool = False) -> Iterator[str]:
    """Get training logs for a model.

    Args:
        model_name: Name of the model
        lines: Number of lines to return
        follow: If True, stream logs as they arrive

    Yields:
        Log lines as strings
    """
    # For streaming logs, we need to use httpx directly
    # as the shared requests module expects JSON responses
    import httpx

    params = {"lines": lines}
    if follow:
        params["follow"] = True

    headers = {"Authorization": f"Bearer {settings.api_key}"}

    with (
        httpx.Client(timeout=300.0) as client,
        client.stream(
            "GET",
            f"{settings.hud_rl_url}/models/{model_name}/training/logs",
            params=params,
            headers=headers,
        ) as response,
    ):
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                yield line
