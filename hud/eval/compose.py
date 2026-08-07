"""Typed Docker Compose data used by runtime adapters."""

from __future__ import annotations

import contextlib
import json
import posixpath
import re
import shlex
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


class ComposeHealthcheck(BaseModel):
    model_config = ConfigDict(extra="allow")

    disable: bool | None = None
    test: list[str] | None = Field(default=None, min_length=1)
    interval: str | None = None
    timeout: str | None = None
    start_period: str | None = None
    start_interval: str | None = None
    retries: int | None = None


class ComposePort(BaseModel):
    model_config = ConfigDict(extra="allow")

    target: int
    protocol: str = "tcp"


class ComposeService(BaseModel):
    """The normalized subset of a Compose service that HUD runtimes execute."""

    model_config = ConfigDict(extra="allow")

    image: str | None = None
    build: str | dict[str, Any] | None = None
    user: str | int | None = None
    environment: dict[str, str] = Field(default_factory=dict)
    entrypoint: list[str] | None = None
    command: list[str] | None = None
    working_dir: str | None = None
    healthcheck: ComposeHealthcheck | None = None
    expose: list[str] = Field(default_factory=list)
    ports: list[ComposePort] = Field(default_factory=list)
    volumes: list[str | dict[str, Any]] = Field(default_factory=list)

    @field_validator("environment", mode="before")
    @classmethod
    def normalize_environment(cls, value: Any) -> Any:
        if isinstance(value, list):
            environment: dict[str, str] = {}
            for item in value:
                if not isinstance(item, str) or "=" not in item:
                    raise ValueError("Compose environment entries must use KEY=VALUE")
                key, _, item_value = item.partition("=")
                environment[key] = item_value
            return environment
        if isinstance(value, dict):
            if any(item is None for item in value.values()):
                raise ValueError("Compose environment values cannot come from the host")
            return {
                key: str(item).lower() if isinstance(item, bool) else str(item)
                for key, item in value.items()
            }
        return value

    @field_validator("entrypoint", "command", mode="before")
    @classmethod
    def normalize_command(cls, value: Any) -> Any:
        return shlex.split(value) if isinstance(value, str) else value

    @field_validator("expose", mode="before")
    @classmethod
    def normalize_expose(cls, value: Any) -> Any:
        return [str(port) for port in value] if isinstance(value, list) else value

    @field_validator("ports", mode="before")
    @classmethod
    def normalize_ports(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        ports: list[Any] = []
        for item in value:
            if isinstance(item, int):
                ports.append({"target": item})
                continue
            if not isinstance(item, str):
                ports.append(item)
                continue
            address, separator, protocol = item.partition("/")
            if separator and protocol not in {"tcp", "udp"}:
                raise ValueError(f"unsupported Compose port protocol {protocol!r}")
            parts = address.rsplit(":", 2)
            target = parts[-1]
            if "-" in target or not target.isdigit():
                raise ValueError(f"unsupported Compose port syntax {item!r}")
            port: dict[str, Any] = {"target": int(target)}
            if separator:
                port["protocol"] = protocol
            if len(parts) >= 2:
                published = parts[-2]
                if published and ("-" in published or not published.isdigit()):
                    raise ValueError(f"unsupported Compose port syntax {item!r}")
                if published:
                    port["published"] = int(published)
            if len(parts) == 3:
                port["host_ip"] = parts[0]
            ports.append(port)
        return ports

    @property
    def argv(self) -> list[str]:
        if self.entrypoint is None or self.command is None:
            raise RuntimeError(f"image defaults were not resolved for {self.image!r}")
        return [*self.entrypoint, *self.command]

    def shell_command(self) -> str:
        command = shlex.join(self.argv)
        if self.working_dir:
            return f"cd {shlex.quote(self.working_dir)} && {command}"
        return command


class ComposeConfig(BaseModel):
    """Normalized Compose data with unknown fields preserved for native runtimes."""

    model_config = ConfigDict(extra="allow")

    name: str | None = None
    services: dict[str, ComposeService]
    networks: dict[str, dict[str, Any] | None] = Field(default_factory=dict)

    @classmethod
    def from_file(cls, path: Path) -> ComposeConfig:
        """Load a self-contained authored Compose document without Docker."""
        source = path.read_text(encoding="utf-8")
        if re.search(r"(?<!\$)\$(?:\{|[A-Za-z_])", source):
            raise ValueError("remote adaptation does not support Compose interpolation")
        raw = yaml.safe_load(source)
        if not isinstance(raw, dict):
            raise ValueError(f"{path.name} is not a Compose document")
        document = raw
        services = document.get("services")
        if "include" in document or (
            isinstance(services, dict)
            and any(
                isinstance(service, dict) and "extends" in service for service in services.values()
            )
        ):
            raise ValueError("remote adaptation does not support Compose include or extends")
        return cls.model_validate(document)

    def with_project_directory(self, directory: str) -> ComposeConfig:
        """Relocate local Compose paths beneath an artifact directory."""
        prefix = posixpath.normpath(directory)
        if prefix == ".." or prefix.startswith(("/", "../")):
            raise ValueError("Compose project directory must stay inside the artifact")

        def relocate(path: str) -> str:
            if path.startswith("/") or "://" in path:
                raise ValueError(f"Compose path {path!r} is outside the artifact")
            source = posixpath.normpath(path)
            if source == ".." or source.startswith("../"):
                raise ValueError(f"Compose path {path!r} escapes its project directory")
            resolved = posixpath.normpath(posixpath.join(prefix, source))
            if resolved == ".." or resolved.startswith("../"):
                raise ValueError(f"Compose path {path!r} escapes the artifact")
            return f"./{resolved.removeprefix('./')}"

        document = self.model_dump(mode="json", exclude_none=True)
        services = document["services"]
        assert isinstance(services, dict)
        for raw_service in services.values():
            assert isinstance(raw_service, dict)
            build = raw_service.get("build")
            if isinstance(build, str):
                raw_service["build"] = relocate(build)
            elif isinstance(build, dict):
                context = build.get("context", ".")
                if not isinstance(context, str):
                    raise ValueError("Compose build context must be a path")
                build["context"] = relocate(context)
                additional_contexts = build.get("additional_contexts")
                if isinstance(additional_contexts, dict):
                    build["additional_contexts"] = {
                        name: (
                            value
                            if not isinstance(value, str) or value.startswith("service:")
                            else relocate(value)
                        )
                        for name, value in additional_contexts.items()
                    }
                elif additional_contexts is not None:
                    raise ValueError("Compose additional_contexts must be a mapping")

            for field in ("env_file", "label_file"):
                files = raw_service.get(field)
                if isinstance(files, str):
                    raw_service[field] = relocate(files)
                elif isinstance(files, list):
                    raw_service[field] = [
                        ({**item, "path": relocate(item["path"])})
                        if isinstance(item, dict) and isinstance(item.get("path"), str)
                        else relocate(item)
                        if isinstance(item, str)
                        else item
                        for item in files
                    ]

            volumes = raw_service.get("volumes")
            if isinstance(volumes, list):
                relocated_volumes: list[Any] = []
                for volume in volumes:
                    if isinstance(volume, str):
                        source, separator, target = volume.partition(":")
                        if separator:
                            if source.startswith("/"):
                                raise ValueError(
                                    f"Compose bind mount {source!r} is outside the artifact"
                                )
                            if source.startswith("."):
                                volume = f"{relocate(source)}:{target}"
                    elif isinstance(volume, dict) and volume.get("type") == "bind":
                        source = volume.get("source")
                        if isinstance(source, str):
                            volume = {**volume, "source": relocate(source)}
                    relocated_volumes.append(volume)
                raw_service["volumes"] = relocated_volumes

        for field in ("configs", "secrets"):
            resources = document.get(field)
            if isinstance(resources, dict):
                for resource in resources.values():
                    if isinstance(resource, dict) and isinstance(resource.get("file"), str):
                        resource["file"] = relocate(resource["file"])
        return ComposeConfig.model_validate(document)


class ComposeProjectRef(BaseModel):
    """Platform reference to a Compose file within an uploaded project."""

    model_config = ConfigDict(extra="forbid")

    compose_path: str


@dataclass(frozen=True, slots=True)
class ComposeSource:
    """One authored or platform-wire Compose runtime source."""

    document: Path | ComposeConfig
    project: Path | ComposeProjectRef | None = None

    def request_payload(self) -> dict[str, Any]:
        if isinstance(self.document, Path):
            document = ComposeConfig.from_file(self.document)
        else:
            document = self.document
        payload: dict[str, Any] = {
            "compose": document.model_dump(mode="json", exclude_none=True),
        }
        if isinstance(self.project, Path):
            if not isinstance(self.document, Path):
                raise ValueError("compose_project as a path requires compose as a path")
            try:
                compose_path = (
                    self.document.resolve().relative_to(self.project.resolve()).as_posix()
                )
            except ValueError:
                raise ValueError("runtime_config.compose must be inside compose_project") from None
            payload["compose_project"] = {"compose_path": compose_path}
        elif self.project is not None:
            payload["compose_project"] = self.project.model_dump(mode="json")
        return payload

    def runnable_path(self, provider: str) -> Path:
        if not isinstance(self.document, Path):
            raise ValueError(f"{provider} requires runtime_config.compose as a local file path")
        return self.document.resolve()


@dataclass(frozen=True, slots=True)
class ComposeLaunchFiles:
    compose: Path
    override: Path
    ports: Path
    archive: Path | None


@dataclass(frozen=True, slots=True)
class ComposeProject:
    """A local Compose project staged with HUD's main-service overrides."""

    compose: Path

    @contextlib.contextmanager
    def stage(
        self,
        published_port: str,
        *,
        seccomp: str | Path,
        service_socket: str | None = None,
        env_vars: Mapping[str, str] | None = None,
        cpu: float | None = None,
        memory_mb: int | None = None,
        gpu_count: int | None = None,
        archive: bool = False,
    ) -> Iterator[ComposeLaunchFiles]:
        main: dict[str, Any] = {
            "security_opt": [f"seccomp={seccomp}", "systempaths=unconfined"],
        }
        if service_socket is not None:
            main["volumes"] = [
                {
                    "type": "bind",
                    "source": service_socket,
                    "target": "/media/hud/docker.sock",
                }
            ]
        if env_vars:
            main["environment"] = dict(env_vars)
        if cpu is not None:
            main["cpus"] = cpu
        if memory_mb is not None:
            main["mem_limit"] = f"{memory_mb}m"
        if gpu_count is not None:
            main["gpus"] = gpu_count

        with tempfile.TemporaryDirectory(prefix="hud-compose-") as directory:
            root = Path(directory)
            override = root / "override.json"
            override.write_text(
                json.dumps({"services": {"main": main}}),
                encoding="utf-8",
            )
            ports = root / "ports.yaml"
            ports.write_text(
                f'services:\n  main:\n    ports: !override ["{published_port}"]\n',
                encoding="utf-8",
            )
            archive_path = None
            if archive:
                archive_path = root / "project.tar.gz"
                with tarfile.open(archive_path, "w:gz") as tar:
                    for entry in self.compose.parent.iterdir():
                        tar.add(entry, arcname=entry.name)
            yield ComposeLaunchFiles(
                compose=self.compose,
                override=override,
                ports=ports,
                archive=archive_path,
            )


__all__ = [
    "ComposeConfig",
    "ComposeHealthcheck",
    "ComposePort",
    "ComposeProject",
    "ComposeProjectRef",
    "ComposeService",
    "ComposeSource",
]
