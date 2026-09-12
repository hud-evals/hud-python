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
    project_id: str | None = None

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
            project_id=str(data["project_id"]) if data.get("project_id") else None,
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


def list_registry_environments(
    platform: PlatformClient,
    *,
    limit: int = 500,
    sort_by: str | None = "date",
) -> list[RegistryEnvironment]:
    environments: list[RegistryEnvironment] = []
    offset = 0
    while True:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if sort_by:
            params["sort_by"] = sort_by
        data = platform.get("/registry", params=params)
        page = [RegistryEnvironment.from_record(item) for item in data["items"]]
        environments.extend(page)
        offset += len(page)
        if offset >= data["total"]:
            return environments
        if not page:
            raise ValueError("Registry API returned an empty page before the reported total")


def resolve_registry_environments(
    platform: PlatformClient,
    ref: str,
) -> list[RegistryEnvironment]:
    """Validate a registry ID against the authenticated platform."""
    try:
        registry_id = str(uuid.UUID(ref))
    except ValueError as exc:
        raise ValueError("Pass an environment ID, or omit it to select interactively") from exc
    environment = get_registry_environment(platform, registry_id)
    return [environment] if environment is not None else []
