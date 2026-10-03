"""Resolve the authored images used by a Harbor environment."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hud.eval.runtime.compose import ComposeConfig

    from .adapt import HarborTask


@dataclass(frozen=True, slots=True)
class ResolvedImages:
    main: dict[str, Any]
    verifier: dict[str, Any]
    peers: dict[str, dict[str, Any]]


class ImageResolutionError(RuntimeError):
    """A Docker command for one environment's images failed or timed out."""


def docker(*args: str, timeout: float | None = None) -> str:
    executable = shutil.which("docker")
    if executable is None:
        raise RuntimeError("Harbor adaptation requires Docker to resolve authored images")
    try:
        result = subprocess.run(
            [executable, *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise ImageResolutionError(
            f"docker {' '.join(args)} timed out after {timeout:g} seconds"
        ) from None
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise ImageResolutionError(f"docker {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def require_docker() -> None:
    """Fail when no Docker daemon is reachable, so outages are not blamed on tasks."""
    try:
        docker("version", "--format", "{{.Server.Version}}", timeout=60)
    except ImageResolutionError as error:
        raise RuntimeError(
            f"Harbor adaptation requires a reachable Docker daemon: {error}"
        ) from error


def inspect_image(image: str) -> dict[str, Any]:
    result = json.loads(docker("image", "inspect", image))
    if not isinstance(result, list) or len(result) != 1:
        raise RuntimeError(f"docker image inspect returned an invalid result for {image!r}")
    config = result[0].get("Config")
    if not isinstance(config, dict):
        raise RuntimeError(f"docker image inspect returned no OCI config for {image!r}")
    return config


def resolve_images(
    source: HarborTask,
    compose_project: ComposeConfig,
    *,
    verifier_image: str,
    peer_services: set[str],
) -> ResolvedImages:
    timeout = source.config.environment.build_timeout_sec
    project_directory = source.path / "environment"
    override: dict[str, dict[str, Any]] = {"services": {}}
    override_services = override["services"]
    main_service = source.compose.services["main"]
    main_override: dict[str, Any] = {"image": source.base_image}
    if main_service.build is None and source.dockerfile.is_file():
        main_override["build"] = {
            "context": ".",
            "dockerfile": source.dockerfile.relative_to(project_directory).as_posix(),
        }
    override_services["main"] = main_override
    for service_name in peer_services:
        service = source.compose.services[service_name]
        target = compose_project.services[service_name].image
        if service.build is not None and target is not None:
            override_services[service_name] = {"image": target}

    # Compose resolves the document adaptation interpolated, never the
    # authored file against the host environment.
    with tempfile.TemporaryDirectory(prefix="hud-harbor-resolve-") as directory:
        document_path = Path(directory) / "compose.json"
        document_path.write_text(
            json.dumps(source.compose.model_dump(mode="json", exclude_none=True)),
            encoding="utf-8",
        )
        override_path = Path(directory) / "override.json"
        override_path.write_text(json.dumps(override), encoding="utf-8")
        command = (
            "compose",
            "--project-name",
            f"hud-adapt-{source.environment_hash}",
            "--project-directory",
            str(project_directory),
            "--file",
            str(document_path),
            "--file",
            str(override_path),
        )
        resolved = {}
        for service_name in ("main", *sorted(peer_services)):
            service = source.compose.services[service_name]
            operation = "build" if service.build is not None else "pull"
            if service_name == "main" and service.build is None:
                operation = "build" if source.dockerfile.is_file() else "pull"
            docker(*command, operation, service_name, timeout=timeout)
            image = (
                source.base_image
                if service_name == "main"
                else compose_project.services[service_name].image
            )
            if image is None:
                raise RuntimeError(f"Docker Compose service {service_name!r} has no image")
            resolved[service_name] = inspect_image(image)
    main = resolved.pop("main")
    peers = resolved

    if source.config.verifier.separate:
        verifier_root = source.path / "tests"
        # The verifier grades what the task produced, so it builds for the task image's platform.
        platform = docker(
            "image", "inspect", "--format", "{{.Os}}/{{.Architecture}}", source.base_image
        )
        try:
            docker(
                "build",
                "--platform",
                platform,
                "--tag",
                verifier_image,
                str(verifier_root),
                timeout=timeout,
            )
        except ImageResolutionError as error:
            host = docker("version", "--format", "{{.Server.Os}}/{{.Server.Arch}}")
            if host.endswith("/amd64"):
                raise
            raise ImageResolutionError(
                f"{error}\nThe docker server builds for {host}. Tasks authored for linux/amd64 "
                "often pin dependencies that publish no wheels for other architectures; set "
                "DOCKER_DEFAULT_PLATFORM=linux/amd64 to adapt the task for one architecture "
                "throughout."
            ) from error
        verifier = inspect_image(verifier_image)
    else:
        verifier = main
    return ResolvedImages(main=main, verifier=verifier, peers=peers)


def image_environment(config: dict[str, Any]) -> dict[str, str]:
    entries = config.get("Env") or []
    if not isinstance(entries, list) or not all(isinstance(entry, str) for entry in entries):
        raise ValueError("OCI image Env must be a list of strings")
    return {
        key: value
        for entry in entries
        for key, separator, value in (entry.partition("="),)
        if separator
    }


def image_ports(config: dict[str, Any], *, image: str) -> set[int]:
    exposed = config.get("ExposedPorts") or {}
    if not isinstance(exposed, dict):
        raise ValueError(f"OCI image ExposedPorts for {image} must be an object")
    return {
        int(port)
        for value in exposed
        if (port := str(value).partition("/")[0]).isdigit()
        and str(value).partition("/")[2] in {"", "tcp"}
    }
