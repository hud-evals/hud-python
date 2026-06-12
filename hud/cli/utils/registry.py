"""Registry environment lookups for the CLI deploy/sync commands."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hud.utils.exceptions import HudRequestError

if TYPE_CHECKING:
    from hud.utils.platform import PlatformClient


@dataclass(frozen=True)
class RegistryEnvironment:
    id: str
    name: str
    version: str = ""

    @classmethod
    def from_record(cls, data: dict[str, Any]) -> RegistryEnvironment:
        """Map one `RegistryDetailResponse` record (version is the latest build's)."""
        env_id = data.get("id")
        if not isinstance(env_id, str) or not env_id:
            raise ValueError("registry environment record needs an id")
        latest_build = data.get("latest_build")
        version = latest_build.get("version") if isinstance(latest_build, dict) else None
        return cls(
            id=env_id,
            name=str(data.get("name") or "unnamed"),
            version=str(version) if version is not None else "",
        )

    @property
    def short_id(self) -> str:
        return self.id[:8]

    @property
    def version_label(self) -> str:
        return f" v{self.version}" if self.version else ""


def get_registry_environment(
    platform: PlatformClient,
    registry_id: str,
) -> RegistryEnvironment | None:
    try:
        data = platform.get(f"/registry/{registry_id}")
    except HudRequestError as e:
        if e.status_code == 404:
            return None
        raise
    if not isinstance(data, dict):
        return None
    return RegistryEnvironment.from_record(data)


def _list_records(platform: PlatformClient, params: dict[str, Any]) -> list[dict[str, Any]]:
    data = platform.get("/registry", params=params)
    items = data.get("items") if isinstance(data, dict) else None
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def list_registry_environments(
    platform: PlatformClient,
    *,
    limit: int = 20,
    sort_by: str | None = "date",
) -> list[RegistryEnvironment]:
    params: dict[str, Any] = {"limit": limit}
    if sort_by:
        params["sort_by"] = sort_by
    return [RegistryEnvironment.from_record(item) for item in _list_records(platform, params)]


def search_registry_environments(
    platform: PlatformClient,
    name: str,
    *,
    limit: int = 5,
) -> list[RegistryEnvironment]:
    records = _list_records(platform, {"search": name, "limit": limit})
    envs = [RegistryEnvironment.from_record(item) for item in records]
    exact = [env for env in envs if env.name == name]
    if exact:
        return exact
    lowered = name.lower()
    return [env for env in envs if lowered in env.name.lower()]


def resolve_registry_environments(
    platform: PlatformClient,
    ref: str,
) -> list[RegistryEnvironment]:
    try:
        uuid.UUID(ref)
        return [RegistryEnvironment(id=ref, name=f"{ref[:8]}...")]
    except ValueError:
        return search_registry_environments(platform, ref)
