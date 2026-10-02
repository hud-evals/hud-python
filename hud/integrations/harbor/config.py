"""What the Harbor adapter hands the controller each image serves.

The adapter writes a :class:`ControllerConfig` as ``config.json`` and binds each
task row's ``task`` argument as a :class:`TaskSpec`; ``env.py`` reads both back.
"""

from __future__ import annotations

import re
import shlex
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from hud.environment import Mount, Peer

if TYPE_CHECKING:
    from hud.eval.runtime.compose import ComposeHealthcheck


class Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(pattern=r"^/")
    destination: str | None = None
    exclude: list[str] = Field(default_factory=list)
    service: str = Field(default="main", min_length=1)

    @model_validator(mode="before")
    @classmethod
    def expand_path(cls, value: Any) -> Any:
        return {"source": value} if isinstance(value, str) else value

    @field_validator("source")
    @classmethod
    def normalize_source(cls, value: str) -> str:
        path = PurePosixPath(value)
        if len(path.parts) == 1 or ".." in path.parts:
            raise ValueError("artifact source must name a path beneath /")
        return str(path)

    @field_validator("destination")
    @classmethod
    def validate_destination(cls, value: str | None) -> str | None:
        if not value:
            return None
        if "\\" in value:
            raise ValueError("artifact destination must use forward slashes")
        path = PurePosixPath(value)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise ValueError("artifact destination must be a relative path")
        if value.rstrip("/") == "manifest.json":
            raise ValueError("artifact destination 'manifest.json' is reserved")
        return value


class Collect(BaseModel):
    model_config = ConfigDict(extra="forbid")

    service: str = Field(default="main", min_length=1)
    command: str = Field(min_length=1)
    timeout_sec: float = Field(default=600.0, gt=0)


class HealthcheckConfig(BaseModel):
    command: str
    interval_sec: float = 5.0
    timeout_sec: float = 30.0
    start_period_sec: float = 0.0
    start_interval_sec: float = 5.0
    retries: int = 3

    @classmethod
    def from_compose(cls, value: ComposeHealthcheck) -> HealthcheckConfig | None:
        if value.disable or value.test in (None, ["NONE"]):
            return None
        test = value.test
        assert test
        if test[0] == "CMD" and len(test) > 1:
            command = shlex.join(str(part) for part in test[1:])
        elif test[0] == "CMD-SHELL" and len(test) == 2:
            command = str(test[1])
        else:
            raise ValueError("Compose main healthcheck test must be CMD or CMD-SHELL")

        def seconds(raw: str | None, default: float) -> float:
            if raw is None:
                return default
            units = {
                "ns": 1e-9,
                "us": 1e-6,
                "µs": 1e-6,
                "ms": 1e-3,
                "s": 1,
                "m": 60,
                "h": 3600,
            }
            parts = re.findall(r"(\d+(?:\.\d+)?)(ns|us|µs|ms|s|m|h)", str(raw))
            if not parts or "".join(number + unit for number, unit in parts) != raw:
                raise ValueError(f"invalid Compose healthcheck duration {raw!r}")
            return sum(float(number) * units[unit] for number, unit in parts)

        return cls(
            command=command,
            interval_sec=seconds(value.interval, 30.0),
            timeout_sec=seconds(value.timeout, 30.0),
            start_period_sec=seconds(value.start_period, 0.0),
            start_interval_sec=seconds(value.start_interval, 5.0),
            retries=value.retries if value.retries is not None else 3,
        )


class Network(BaseModel):
    """Whether a phase has a network, and the hosts it may reach (``*`` for any)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    allowed_hosts: list[str]


class PhasePolicy(BaseModel):
    """Who a phase runs as (``None`` is root) and what it may reach."""

    model_config = ConfigDict(extra="forbid")

    user: str | int | None
    network: Network
    env: dict[str, str]


class EnvironmentPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env: dict[str, str]
    network: Network
    healthcheck: HealthcheckConfig | None


class VerifierImage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workdir: str
    env: dict[str, str]


class ControllerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    mounts: list[Mount]
    workdir: str
    image_user: str | int | None
    image_env: dict[str, str]
    entrypoint: list[str]
    ports: list[int]
    verifier_root: str | None
    verifier_image: VerifierImage
    environment: EnvironmentPolicy
    agent: PhasePolicy
    verifier: PhasePolicy
    capabilities: list[dict[str, Any]]
    local_aliases: list[str]
    peers: list[Peer]


class TaskSpec(BaseModel):
    """One Harbor task's grading contract, bound as the ``task`` template argument."""

    model_config = ConfigDict(extra="forbid")

    id: str
    description: str
    verifier_timeout: float
    separate_verifier: bool
    collect: list[Collect]
    artifacts: list[Artifact]
