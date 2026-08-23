"""Typed Docker Compose data used by runtime adapters."""

from __future__ import annotations

import contextlib
import json
import os
import posixpath
import re
import shlex
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from dotenv import dotenv_values
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializationInfo,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


_COMPOSE_VARIABLE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class ComposeUnboundVariableError(ValueError):
    """A Compose variable depends on values outside the packaged project."""


def _interpolate_compose_value(
    value: str,
    environment: Mapping[str, str],
    *,
    escape_dollars: bool = True,
) -> str:
    result: list[str] = []
    index = 0
    while index < len(value):
        marker = value.find("$", index)
        if marker < 0:
            result.append(value[index:])
            break
        result.append(value[index:marker])
        if marker + 1 >= len(value):
            result.append("$$")
            break
        following = value[marker + 1]
        if following == "$":
            result.append("$$")
            index = marker + 2
            continue
        if following == "{":
            depth = 1
            end = marker + 2
            while end < len(value) and depth:
                if value.startswith("${", end):
                    depth += 1
                    end += 2
                    continue
                if value[end] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                end += 1
            if depth:
                raise ValueError("invalid Compose interpolation: unclosed variable")
            expression = value[marker + 2 : end]
            result.append(_resolve_compose_variable(expression, environment))
            index = end + 1
            continue
        match = _COMPOSE_VARIABLE.match(value, marker + 1)
        if match is None:
            result.append("$$")
            index = marker + 1
            continue
        name = match.group()
        if name not in environment:
            raise ComposeUnboundVariableError(
                f"Compose variable {name!r} is not set by the project .env"
            )
        result.append(environment[name].replace("$", "$$"))
        index = match.end()
    resolved = "".join(result)
    return resolved if escape_dollars else resolved.replace("$$", "$")


def _resolve_compose_variable(expression: str, environment: Mapping[str, str]) -> str:
    match = _COMPOSE_VARIABLE.match(expression)
    if match is None:
        raise ValueError(f"invalid Compose interpolation expression {expression!r}")
    name = match.group()
    suffix = expression[match.end() :]
    value = environment.get(name)
    escaped = value.replace("$", "$$") if value is not None else ""
    if not suffix:
        if value is None:
            raise ComposeUnboundVariableError(
                f"Compose variable {name!r} is not set by the project .env"
            )
        return escaped

    operator = next(
        (item for item in (":-", ":?", ":+", "-", "?", "+") if suffix.startswith(item)), None
    )
    if operator is None:
        raise ValueError(f"invalid Compose interpolation expression {expression!r}")
    operand = suffix[len(operator) :]
    is_set = value is not None
    is_nonempty = is_set and value != ""
    if operator == ":-":
        return escaped if is_nonempty else _interpolate_compose_value(operand, environment)
    if operator == "-":
        return escaped if is_set else _interpolate_compose_value(operand, environment)
    if operator == ":+":
        return _interpolate_compose_value(operand, environment) if is_nonempty else ""
    if operator == "+":
        return _interpolate_compose_value(operand, environment) if is_set else ""
    if (operator == ":?" and not is_nonempty) or (operator == "?" and not is_set):
        detail = operand or f"Compose variable {name!r} is required"
        raise ComposeUnboundVariableError(detail)
    return escaped


def _interpolate_compose_node(
    node: Node,
    environment: Mapping[str, str],
    seen: set[int],
) -> None:
    if id(node) in seen:
        return
    seen.add(id(node))
    if isinstance(node, MappingNode):
        for _, value in node.value:
            _interpolate_compose_node(value, environment, seen)
    elif isinstance(node, SequenceNode):
        for value in node.value:
            _interpolate_compose_node(value, environment, seen)
    elif isinstance(node, ScalarNode) and node.tag == "tag:yaml.org,2002:str":
        node.value = (
            node.value.replace("$", "$$")
            if node.style == "'"
            else _interpolate_compose_value(node.value, environment)
        )


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
    network_mode: str | None = None
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

    @property
    def tcp_ports(self) -> set[int]:
        ports: set[int] = set()
        for exposed in self.expose:
            value, _, protocol = exposed.partition("/")
            if value.isdigit() and protocol in {"", "tcp"}:
                ports.add(int(value))
        ports.update(port.target for port in self.ports if port.protocol == "tcp")
        return ports

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

    def network_owner(self, service: str) -> str:
        """Service whose network namespace and published ports *service* uses."""
        seen: set[str] = set()
        current = service
        while True:
            if current in seen:
                raise ValueError(f"Compose network_mode service cycle includes {current!r}")
            seen.add(current)
            try:
                mode = self.services[current].network_mode
            except KeyError:
                raise ValueError(
                    f"Compose network_mode references unknown service {current!r}"
                ) from None
            if mode is None or not mode.startswith("service:"):
                return current
            current = mode.removeprefix("service:")

    @classmethod
    def from_file(cls, path: Path) -> ComposeConfig:
        """Load a self-contained authored Compose document without Docker."""
        source = path.read_text(encoding="utf-8")
        environment: dict[str, str] = {}
        for key, value in dotenv_values(path.parent / ".env", interpolate=False).items():
            if value is not None:
                environment[key] = _interpolate_compose_value(
                    value,
                    environment,
                    escape_dollars=False,
                )
        loader = yaml.SafeLoader(source)
        try:
            node = loader.get_single_node()
            if node is None:
                raise ValueError(f"{path.name} is not a Compose document")
            _interpolate_compose_node(node, environment, set())
            raw = loader.construct_document(node)
        finally:
            loader.dispose()
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
class ComposeLaunchFiles:
    compose: Path
    project_directory: Path
    override: Path
    ports: Path
    archive: Path | None


class ComposeProject(BaseModel):
    """A Compose recipe and the project data it may need at runtime."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document: Path | ComposeConfig
    root: Path | ComposeProjectRef | None = None
    service_access: bool | None = None

    @field_validator("document", "root", mode="before")
    @classmethod
    def resolve_local_path(cls, value: Any, info: ValidationInfo) -> Any:
        base = (info.context or {}).get("base_path")
        if isinstance(value, str) and isinstance(base, Path):
            return (base / value).resolve()
        return value

    @model_validator(mode="after")
    def validate_source(self) -> ComposeProject:
        if self.root is None:
            return self
        if isinstance(self.root, Path) != isinstance(self.document, Path):
            raise ValueError("Compose source and project root must use the same form")
        if isinstance(self.root, Path):
            assert isinstance(self.document, Path)
            try:
                self.document.resolve().relative_to(self.root.resolve())
            except ValueError:
                raise ValueError("Compose file must be inside its project root") from None
        return self

    @field_serializer("document", when_used="json")
    def serialize_document(
        self,
        document: Path | ComposeConfig,
        info: SerializationInfo,
    ) -> dict[str, Any] | str:
        base = (info.context or {}).get("base_path")
        if isinstance(document, Path) and isinstance(base, Path):
            return os.path.relpath(document.resolve(), base)
        config = ComposeConfig.from_file(document) if isinstance(document, Path) else document
        return config.model_dump(mode="json", exclude_none=True)

    @field_serializer("root", when_used="json")
    def serialize_root(
        self,
        root: Path | ComposeProjectRef | None,
        info: SerializationInfo,
    ) -> dict[str, str] | str | None:
        if root is None:
            return None
        if isinstance(root, ComposeProjectRef):
            return root.model_dump(mode="json")
        base = (info.context or {}).get("base_path")
        if isinstance(base, Path):
            return os.path.relpath(root.resolve(), base)
        assert isinstance(self.document, Path)
        return {
            "compose_path": self.document.resolve().relative_to(root.resolve()).as_posix(),
        }

    @contextlib.contextmanager
    def stage(
        self,
        published_port: str,
        *,
        port_service: str = "main",
        seccomp: str | Path,
        service_socket: str | None = None,
        env_vars: Mapping[str, str] | None = None,
        cpu: float | None = None,
        memory_mb: int | None = None,
        gpu_count: int | None = None,
        archive: bool = False,
    ) -> Iterator[ComposeLaunchFiles]:
        if not isinstance(self.document, Path):
            raise ValueError("Compose project is not available on the local filesystem")
        compose = self.document.resolve()
        main: dict[str, Any] = {
            "security_opt": [
                f"seccomp={seccomp}",
                "systempaths=unconfined",
                "apparmor=unconfined",
            ],
            "volumes": [
                {
                    "type": "volume",
                    "source": "hud-runtime-sessions",
                    "target": target,
                }
                for target in ("/runtime/sessions", "/media/hud/sessions")
            ],
        }
        if service_socket is not None:
            main["volumes"].extend(
                {
                    "type": "bind",
                    "source": service_socket,
                    "target": target,
                }
                for target in ("/var/run/docker.sock", "/media/hud/docker.sock")
            )
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
            normalized = root / "compose.json"
            normalized.write_text(
                json.dumps(
                    ComposeConfig.from_file(compose).model_dump(mode="json", exclude_none=True)
                ),
                encoding="utf-8",
            )
            override = root / "override.json"
            override.write_text(
                json.dumps(
                    {
                        "services": {"main": main},
                        "volumes": {"hud-runtime-sessions": {}},
                    }
                ),
                encoding="utf-8",
            )
            ports = root / "ports.yaml"
            ports.write_text(
                f'services:\n  {port_service}:\n    ports: !override ["{published_port}"]\n',
                encoding="utf-8",
            )
            archive_path = None
            if archive:
                archive_path = root / "project.tar.gz"
                project_root = (
                    self.root.resolve() if isinstance(self.root, Path) else compose.parent
                )
                compose_path = compose.relative_to(project_root).as_posix()

                def omit_authored_compose(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
                    return None if info.name == compose_path else info

                with tarfile.open(archive_path, "w:gz") as tar:
                    for entry in project_root.iterdir():
                        tar.add(entry, arcname=entry.name, filter=omit_authored_compose)
                    tar.add(normalized, arcname=compose_path)
            yield ComposeLaunchFiles(
                compose=normalized,
                project_directory=compose.parent,
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
]
