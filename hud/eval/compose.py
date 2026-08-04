"""Typed Docker Compose data used by runtime adapters."""

from __future__ import annotations

import shlex
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

DockerCommand = Callable[..., Awaitable[tuple[str, str]]]

if TYPE_CHECKING:
    from pathlib import Path


class ImageConfig(BaseModel):
    """OCI image defaults that Compose applies when service fields are unset."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    user: str | None = Field(default=None, alias="User")
    working_dir: str | None = Field(default=None, alias="WorkingDir")
    entrypoint: list[str] | None = Field(default=None, alias="Entrypoint")
    command: list[str] | None = Field(default=None, alias="Cmd")
    environment: list[str] = Field(default_factory=list, alias="Env")
    exposed_ports: dict[str, Any] = Field(default_factory=dict, alias="ExposedPorts")

    @classmethod
    async def inspect(cls, image: str, docker: DockerCommand) -> ImageConfig:
        output, _ = await docker("image", "inspect", "--format", "{{json .Config}}", image)
        return cls.model_validate_json(output)

    @classmethod
    async def inspect_registry(
        cls,
        image: str,
        docker: DockerCommand,
        *,
        platform: str = "linux/amd64",
    ) -> ImageConfig:
        template = f'{{{{json (index .Image "{platform}").Config}}}}'
        output, _ = await docker(
            "buildx",
            "imagetools",
            "inspect",
            "--format",
            template,
            image,
        )
        return cls.model_validate_json(output)


class ComposeHealthcheck(BaseModel):
    model_config = ConfigDict(extra="allow")

    test: list[str] | None = None


class ComposePort(BaseModel):
    model_config = ConfigDict(extra="allow")

    target: int
    protocol: str = "tcp"


class ComposeService(BaseModel):
    """The normalized subset of a Compose service that HUD runtimes execute."""

    model_config = ConfigDict(extra="allow")

    image: str | None = None
    build: str | dict[str, Any] | None = None
    environment: dict[str, str] = Field(default_factory=dict)
    entrypoint: list[str] | None = None
    command: list[str] | None = None
    working_dir: str | None = None
    healthcheck: ComposeHealthcheck | None = None
    expose: list[str] = Field(default_factory=list)
    ports: list[ComposePort] = Field(default_factory=list)

    def with_image(self, image: str, config: ImageConfig) -> ComposeService:
        entrypoint = (config.entrypoint or []) if self.entrypoint is None else self.entrypoint
        if self.command is not None:
            command = self.command
        elif self.entrypoint is not None:
            command = []
        else:
            command = config.command or []
        return self.model_copy(
            update={
                "image": image,
                "entrypoint": entrypoint,
                "command": command,
                "working_dir": self.working_dir or config.working_dir or "/",
            }
        )

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
    async def load(
        cls,
        *files: Path,
        docker: DockerCommand,
        project_directory: Path | None = None,
        project_name: str | None = None,
    ) -> ComposeConfig:
        if not files:
            raise ValueError("ComposeConfig.load requires at least one file")
        command = ["compose"]
        if project_name is not None:
            command.extend(("--project-name", project_name))
        if project_directory is not None:
            command.extend(("--project-directory", str(project_directory)))
        for file in files:
            command.extend(("--file", str(file)))
        output, _ = await docker(*command, "config", "--format", "json")
        return cls.model_validate_json(output)

    async def resolve_registry_images(
        self,
        docker: DockerCommand,
        *,
        platform: str = "linux/amd64",
    ) -> ComposeConfig:
        images: dict[str, ImageConfig] = {}
        services: dict[str, ComposeService] = {}
        for name, service in self.services.items():
            if service.image is None:
                raise ValueError(f"Compose service {name!r} requires an image")
            if service.entrypoint is None or service.command is None or service.working_dir is None:
                if service.image not in images:
                    images[service.image] = await ImageConfig.inspect_registry(
                        service.image,
                        docker,
                        platform=platform,
                    )
                service = service.with_image(service.image, images[service.image])
            services[name] = service
        return self.model_copy(update={"services": services})


__all__ = ["ComposeConfig", "ComposeHealthcheck", "ComposePort", "ComposeService", "ImageConfig"]
