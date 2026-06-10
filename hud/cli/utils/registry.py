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
        env_id = data.get("id")
        if not isinstance(env_id, str) or not env_id:
            raise ValueError("registry environment record needs an id")
        display = data.get("name_display") or data.get("name") or "unnamed"
        version = data.get("latest_version") or ""
        return cls(id=env_id, name=str(display), version=str(version) if version else "")

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
        data = platform.get(f"/registry/envs/{registry_id}")
    except HudRequestError as e:
        if e.status_code == 404:
            return None
        raise
    if not isinstance(data, dict):
        return None
    return RegistryEnvironment.from_record(data)


def list_registry_environments(
    platform: PlatformClient,
    *,
    limit: int = 20,
    sort_by: str | None = "updated_at",
) -> list[RegistryEnvironment]:
    params: dict[str, Any] = {"limit": limit}
    if sort_by:
        params["sort_by"] = sort_by
    data = platform.get("/registry/envs", params=params)
    return [RegistryEnvironment.from_record(item) for item in data if isinstance(item, dict)]


def search_registry_environments(
    platform: PlatformClient,
    name: str,
    *,
    limit: int = 5,
) -> list[RegistryEnvironment]:
    data = platform.get("/registry/envs", params={"search": name, "limit": limit})
    envs = [RegistryEnvironment.from_record(item) for item in data if isinstance(item, dict)]
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
