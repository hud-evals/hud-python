"""Shared evalset resolution utilities used by ``hud sync`` and ``hud eval``."""

from __future__ import annotations

import logging
from typing import Any
from urllib import parse

import httpx

LOGGER = logging.getLogger(__name__)


def resolve_taskset_id(
    name_or_id: str,
    api_url: str,
    headers: dict[str, str],
    *,
    create: bool = True,
) -> tuple[str, str, bool]:
    """Resolve a taskset name to its UUID.

    Args:
        create: If True (default), creates the evalset if it doesn't exist.
            Set to False for read-only operations like ``hud eval``.

    Returns (evalset_id, evalset_name, created).
    Returns ("", name, False) if not found and create=False.
    """
    try:
        import uuid as _uuid

        _uuid.UUID(name_or_id)
        return name_or_id, name_or_id, False
    except ValueError:
        pass

    if create:
        response = httpx.post(
            f"{api_url}/tasks/resolve-evalset",
            json={"name": name_or_id},
            headers=headers,
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        return (
            str(data.get("evalset_id", "")),
            str(data.get("name", name_or_id)),
            bool(data.get("created", False)),
        )

    response = httpx.get(
        f"{api_url}/tasks/evalset/{parse.quote(name_or_id, safe='')}",
        headers=headers,
        timeout=30.0,
    )
    if response.status_code == 404:
        return "", name_or_id, False
    response.raise_for_status()
    data = response.json()
    return str(data.get("evalset_id", "")), str(data.get("evalset_name", name_or_id)), False


def fetch_remote_tasks(
    evalset_id: str,
    api_url: str,
    headers: dict[str, str],
) -> list[dict[str, Any]]:
    """Fetch remote tasks for an evalset by UUID."""
    response = httpx.get(
        f"{api_url}/tasks/evalsets/{evalset_id}/tasks-by-id",
        headers=headers,
        timeout=30.0,
    )
    if response.status_code == 404:
        return []
    response.raise_for_status()
    data = response.json()
    tasks_payload = data.get("tasks") or {}
    if not isinstance(tasks_payload, dict):
        return []
    return [entry for entry in tasks_payload.values() if isinstance(entry, dict)]
