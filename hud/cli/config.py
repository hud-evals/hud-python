from __future__ import annotations

import os
import tempfile
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlsplit
from uuid import UUID

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, StrictBool

if TYPE_CHECKING:
    from hud.utils.platform import PlatformClient


CONFIG_PATH = Path(".hud") / "config.json"


class DirectoryLink(BaseModel):
    model_config = ConfigDict(extra="forbid")

    registry_id: UUID | None = None
    taskset_id: UUID | None = None
    project_id: UUID | None = None
    sync_env: dict[UUID, StrictBool] = Field(default_factory=dict)


class AuthScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    origin: str
    user_id: UUID
    team_id: UUID

    @classmethod
    def resolve(cls, platform: PlatformClient) -> AuthScope:
        identity = platform.get("/auth/me")
        url = urlsplit(platform.api_url)
        if url.scheme not in {"http", "https"} or not url.hostname:
            raise ValueError("HUD API URL must be an HTTP origin")
        default_port = 443 if url.scheme == "https" else 80
        port = "" if url.port in (None, default_port) else f":{url.port}"
        return cls(
            origin=f"{url.scheme}://{url.hostname}{port}",
            user_id=identity["user_id"],
            team_id=identity["team_id"],
        )


class WorkspaceLink(BaseModel):
    """CLI-owned binding for one working copy. Not a human-edited settings file."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    scope: AuthScope
    registry_id: UUID | None = None
    taskset_id: UUID | None = None
    project_id: UUID | None = None
    sync_env: dict[UUID, StrictBool] = Field(default_factory=dict)

    def as_link(self) -> DirectoryLink:
        return DirectoryLink(
            registry_id=self.registry_id,
            taskset_id=self.taskset_id,
            project_id=self.project_id,
            sync_env=self.sync_env,
        )


class DirectoryState:
    def __init__(self, scope: AuthScope, directory: str | Path = ".") -> None:
        self.scope = scope
        self.directory = str(Path(directory).expanduser().resolve())

    @property
    def path(self) -> Path:
        return Path(self.directory) / CONFIG_PATH

    def load(self) -> DirectoryLink:
        stored = self._read()
        return stored.as_link() if stored is not None else DirectoryLink()

    def update(self, changes: DirectoryLink) -> bool:
        stored = self._read()
        current = stored.as_link() if stored is not None else DirectoryLink()
        values = current.model_dump()
        updates = changes.model_dump(exclude_unset=True)
        if "sync_env" in updates:
            updates["sync_env"] = {**current.sync_env, **changes.sync_env}
        updated = DirectoryLink.model_validate({**values, **updates})
        if updated == current:
            return False
        _write_config_file(
            self.path,
            WorkspaceLink(scope=self.scope, **updated.model_dump()).model_dump_json(indent=2)
            + "\n",
        )
        return True

    def _read(self) -> WorkspaceLink | None:
        path = self.path
        if not path.exists():
            return None
        try:
            stored = WorkspaceLink.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ValueError(f"Invalid HUD workspace link at {path}: {exc}") from exc
        if stored.scope != self.scope:
            raise ValueError(
                f"Workspace link at {path} is for {stored.scope.origin} "
                f"team {stored.scope.team_id}, not the current credentials"
            )
        return stored


def _write_config_file(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".config-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(contents)
            stream.flush()
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def get_config_dir() -> Path:
    return Path.home() / ".hud"


def get_user_env_path() -> Path:
    return get_config_dir() / ".env"


def ensure_config_dir() -> Path:
    config_dir = get_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def parse_key_value(item: str) -> tuple[str, str] | None:
    key, sep, value = item.partition("=")
    key = key.strip()
    if not sep or not key:
        return None
    return key, value.strip()


def parse_env_file(contents: str) -> dict[str, str]:
    return {
        key: value
        for key, value in dotenv_values(stream=StringIO(contents), interpolate=False).items()
        if value is not None
    }


def render_env_file(env: dict[str, str]) -> str:
    header = [
        "# HUD CLI persistent environment file",
        "# Keys set via `hud set KEY=VALUE`",
        "# This file is read after process env and project .env",
        "# so project overrides take precedence over these defaults.",
        "",
    ]
    body = []
    for key, value in sorted(env.items()):
        quoted = value.replace("\\", "\\\\").replace("'", "\\'")
        body.append(f"{key}='{quoted}'")
    return "\n".join([*header, *body, ""])


def load_env_file(path: Path | None = None) -> dict[str, str]:
    env_path = path or get_user_env_path()
    if not env_path.exists():
        return {}
    contents = env_path.read_text(encoding="utf-8")
    return parse_env_file(contents)


def save_env_file(env: dict[str, str], path: Path | None = None) -> Path:
    env_path = path or get_user_env_path()
    _write_config_file(env_path, render_env_file(env))
    return env_path


def set_env_values(values: dict[str, str]) -> Path:
    current = load_env_file()
    current.update(values)
    return save_env_file(current)
