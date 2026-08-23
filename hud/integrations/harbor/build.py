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

COMPOSE_FILENAME = "docker-compose.yaml"


@dataclass(frozen=True, slots=True)
class ResolvedImages:
    main: dict[str, Any]
    verifier: dict[str, Any]
    peers: dict[str, dict[str, Any]]


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
        raise TimeoutError(f"Docker command timed out after {timeout:g} seconds") from None
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"docker {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


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
    compose_project: ComposeConfig | None,
    *,
    verifier_image: str,
    peer_services: set[str],
) -> ResolvedImages:
    timeout = source.config.environment.build_timeout_sec
    if source.compose is None:
        if source.dockerfile.is_file():
            docker(
                "build",
                "--tag",
                source.base_image,
                "--file",
                str(source.dockerfile),
                str(source.dockerfile.parent),
                timeout=timeout,
            )
        else:
            docker("pull", source.base_image, timeout=timeout)
        main = inspect_image(source.base_image)
        peers: dict[str, dict[str, Any]] = {}
    else:
        assert compose_project is not None
        compose_file = source.path / "environment" / COMPOSE_FILENAME
        override: dict[str, dict[str, Any]] = {"services": {}}
        override_services = override["services"]
        main_service = source.compose.services["main"]
        if main_service.build is not None:
            override_services["main"] = {"image": source.base_image}
        elif main_service.image is None:
            override_services["main"] = (
                {
                    "image": source.base_image,
                    "build": {
                        "context": ".",
                        "dockerfile": source.dockerfile.relative_to(compose_file.parent).as_posix(),
                    },
                }
                if source.dockerfile.is_file()
                else {"image": source.base_image}
            )
        for service_name in peer_services:
            service = source.compose.services[service_name]
            target = compose_project.services[service_name].image
            if service.build is not None and target is not None:
                override_services[service_name] = {"image": target}

        with tempfile.TemporaryDirectory(prefix="hud-harbor-resolve-") as directory:
            override_path = Path(directory) / "compose.json"
            override_path.write_text(json.dumps(override), encoding="utf-8")
            command = (
                "compose",
                "--project-name",
                f"hud-adapt-{source.environment_hash}",
                "--project-directory",
                str(compose_file.parent),
                "--file",
                str(compose_file),
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
        docker("build", "--tag", verifier_image, str(verifier_root), timeout=timeout)
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
