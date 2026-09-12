from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlsplit
from uuid import UUID

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

if os.name == "nt":
    import msvcrt
else:
    import fcntl

if TYPE_CHECKING:
    from collections.abc import Iterator

    from hud.utils.platform import PlatformClient


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
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.path not in {"", "/"}
            or url.query
            or url.fragment
        ):
            raise ValueError("HUD API URL must be an HTTP origin")
        port = url.port
        if port == (443 if url.scheme == "https" else 80):
            port = None
        host = f"[{url.hostname}]" if ":" in url.hostname else url.hostname
        origin = f"{url.scheme}://{host}" + (f":{port}" if port else "")
        return cls(origin=origin, user_id=identity["user_id"], team_id=identity["team_id"])


class DirectoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: AuthScope
    directory: str
    link: DirectoryLink

    @model_validator(mode="after")
    def absolute_directory(self) -> DirectoryConfig:
        path = Path(self.directory)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("Directory links require canonical absolute paths")
        return self


class CLIConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    directories: list[DirectoryConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_directories(self) -> CLIConfig:
        keys = [(entry.scope, entry.directory) for entry in self.directories]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate scoped directory entries")
        return self


def load_config() -> CLIConfig:
    path = get_config_dir() / "config.json"
    if not path.exists():
        return CLIConfig()
    try:
        return CLIConfig.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"Invalid HUD configuration at {path}: {exc}") from exc


@contextmanager
def _config_lock() -> Iterator[None]:
    path = ensure_config_dir() / "config.lock"
    with path.open("a+b") as lock:
        if os.name == "nt":
            if lock.seek(0, os.SEEK_END) == 0:
                lock.write(b"\0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


class DirectoryState:
    def __init__(self, scope: AuthScope, directory: str | Path = ".") -> None:
        self.scope = scope
        self.directory = str(Path(directory).expanduser().resolve())

    def load(self) -> DirectoryLink:
        for entry in load_config().directories:
            if entry.scope == self.scope and entry.directory == self.directory:
                return entry.link
        return DirectoryLink()

    def update(self, changes: DirectoryLink) -> bool:
        with _config_lock():
            config = load_config()
            entry = next(
                (
                    entry
                    for entry in config.directories
                    if entry.scope == self.scope and entry.directory == self.directory
                ),
                None,
            )
            if entry is None:
                entry = DirectoryConfig(
                    scope=self.scope, directory=self.directory, link=DirectoryLink()
                )
                config.directories.append(entry)
            values = entry.link.model_dump()
            updates = changes.model_dump(exclude_unset=True)
            if "sync_env" in updates:
                updates["sync_env"] = {**entry.link.sync_env, **changes.sync_env}
            updated = DirectoryLink.model_validate({**values, **updates})
            if updated == entry.link:
                return False
            entry.link = updated
            path = get_config_dir() / "config.json"
            _write_config_file(path, config.model_dump_json(indent=2) + "\n")
            return True


def _write_config_file(path: Path, contents: str) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".config-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def get_config_dir() -> Path:
    """Return the base HUD config directory in the user's home.

    Uses ~/.hud across platforms for consistency with existing registry data.
    """
    return Path.home() / ".hud"


def get_user_env_path() -> Path:
    """Return the path to the persistent user-level env file (~/.hud/.env)."""
    return get_config_dir() / ".env"


def ensure_config_dir() -> Path:
    """Ensure the HUD config directory exists and return it."""
    config_dir = get_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def parse_key_value(item: str) -> tuple[str, str] | None:
    """Split one ``KEY=VALUE`` string into ``(key, value)``.

    Returns ``None`` for malformed input (no ``=`` or empty key); each caller
    decides whether that's a warning, an error, or a skip.
    """
    key, sep, value = item.partition("=")
    key = key.strip()
    if not sep or not key:
        return None
    return key, value.strip()


def parse_env_file(contents: str) -> dict[str, str]:
    """Read dotenv syntax without expanding variable references."""
    return {
        key: value
        for key, value in dotenv_values(stream=StringIO(contents), interpolate=False).items()
        if value is not None
    }


def render_env_file(env: dict[str, str]) -> str:
    """Render a dict of env values to KEY=VALUE lines with a header."""
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
    """Load env assignments from the given path (defaults to ~/.hud/.env)."""
    env_path = path or get_user_env_path()
    if not env_path.exists():
        return {}
    contents = env_path.read_text(encoding="utf-8")
    return parse_env_file(contents)


def save_env_file(env: dict[str, str], path: Path | None = None) -> Path:
    """Write env assignments to the given path and return the path."""
    ensure_config_dir()
    env_path = path or get_user_env_path()
    rendered = render_env_file(env)
    _write_config_file(env_path, rendered)
    return env_path


def set_env_values(values: dict[str, str]) -> Path:
    """Persist provided KEY=VALUE pairs into ~/.hud/.env and return the path."""
    with _config_lock():
        current = load_env_file()
        current.update(values)
        return save_env_file(current)
