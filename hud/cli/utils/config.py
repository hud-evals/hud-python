from __future__ import annotations

from pathlib import Path


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


def parse_env_file(contents: str) -> dict[str, str]:
    """Parse simple KEY=VALUE lines into a dict.

    - Ignores blank lines and lines starting with '#'.
    - Does not perform variable substitution or quoting.
    """
    data: dict[str, str] = {}
    for raw_line in contents.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            data[key] = value
    return data


def render_env_file(env: dict[str, str]) -> str:
    """Render a dict of env values to KEY=VALUE lines with a header."""
    header = [
        "# HUD CLI persistent environment file",
        "# Keys set via `hud set KEY=VALUE`",
        "# This file is read after process env and project .env",
        "# so project overrides take precedence over these defaults.",
        "",
    ]
    body = [f"{key}={env[key]}" for key in sorted(env.keys())]
    return "\n".join([*header, *body, ""])


def load_env_file(path: Path | None = None) -> dict[str, str]:
    """Load env assignments from the given path (defaults to ~/.hud/.env)."""
    env_path = path or get_user_env_path()
    if not env_path.exists():
        return {}
    try:
        contents = env_path.read_text(encoding="utf-8")
    except Exception:
        return {}
    return parse_env_file(contents)


def save_env_file(env: dict[str, str], path: Path | None = None) -> Path:
    """Write env assignments to the given path and return the path."""
    ensure_config_dir()
    env_path = path or get_user_env_path()
    rendered = render_env_file(env)
    env_path.write_text(rendered, encoding="utf-8")
    return env_path


def set_env_values(values: dict[str, str]) -> Path:
    """Persist provided KEY=VALUE pairs into ~/.hud/.env and return the path."""
    current = load_env_file()
    current.update(values)
    return save_env_file(current)
