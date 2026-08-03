"""Platform builds for an environment's build context: the wire sequence.

Deploying is upload → trigger → await: the context goes to object storage, the
platform builds the image and introspects what it serves, and the build's lock
comes back. This module owns that exchange and nothing else — the counterpart
to :mod:`hud.eval.sync`, which owns the row exchange.

What a *directory* is called, which environment it is linked to, and what a
person watching is told are the CLI's concerns (:mod:`hud.cli.deploy`). Core
is handed a name and a context and reports what the platform did; it reads no
project configuration and writes nothing into the caller's tree.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 - part of the public signature
from typing import TYPE_CHECKING, Any

import httpx

from hud.utils.build_context import create_build_context_tarball
from hud.utils.exceptions import HudException

if TYPE_CHECKING:
    from hud.utils.platform import PlatformClient

#: Build states the platform does not move on from.
TERMINAL_STATUSES = frozenset({"SUCCEEDED", "FAILED", "STOPPED", "TIMED_OUT"})


class DeployError(HudException):
    """A deploy that could not be carried out (as opposed to a build that ran
    and failed — that is a :class:`BuildOutcome` whose status is not
    ``SUCCEEDED``)."""


@dataclass(frozen=True, slots=True)
class BuildOutcome:
    """What one platform build did."""

    build_id: str
    registry_id: str
    status: str
    version: str | None = None
    image_uri: str | None = None
    registry_name: str | None = None
    #: The built image's manifest — the tasks it serves and their arg schemas.
    lock: dict[str, Any] | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == "SUCCEEDED"


async def create_build(platform: PlatformClient) -> tuple[str, str]:
    """Reserve a build and its upload URL: ``(upload_url, build_id)``."""
    data = await platform.apost("/builds/upload-url")
    return data["upload_url"], data["build_id"]


async def upload_context(upload_url: str, tarball: Path, *, deadline: float = 300.0) -> None:
    """PUT the context tarball to the presigned URL (object storage, not the API)."""
    content = await asyncio.to_thread(tarball.read_bytes)
    async with httpx.AsyncClient(timeout=deadline) as client:
        response = await client.put(
            upload_url,
            content=content,
            headers={"Content-Type": "application/gzip"},
        )
        response.raise_for_status()


async def trigger_build(
    platform: PlatformClient,
    *,
    build_id: str,
    name: str,
    registry_id: str | None = None,
    env_vars: dict[str, str] | None = None,
    build_args: dict[str, str] | None = None,
    build_secrets: dict[str, str] | None = None,
    runtime: str | None = None,
    runtime_config: dict[str, Any] | None = None,
    no_cache: bool = False,
) -> tuple[str, str]:
    """Start the build; returns ``(build_id, registry_id)``.

    The platform resolves the registry by *name* (get-or-rebuild), so an
    existing environment with this name gets a new version rather than a
    second registry entry.
    """
    payload: dict[str, Any] = {
        "source": "direct",
        "build_id": build_id,
        "name": name,
        "no_cache": no_cache,
    }
    payload.update(
        {
            key: value
            for key, value in (
                ("registry_id", registry_id),
                ("runtime_provider", runtime),
                ("runtime_config", runtime_config),
                ("environment_variables", env_vars),
                ("build_args", build_args),
                ("build_secrets", build_secrets),
            )
            if value
        }
    )
    data = await platform.apost("/builds/trigger", json=payload)
    return data["id"], data["registry_id"]


async def build_status(platform: PlatformClient, build_id: str) -> dict[str, Any]:
    """The build's current status document."""
    return await platform.aget(f"/builds/{build_id}/status")


async def await_build(
    platform: PlatformClient,
    build_id: str,
    *,
    poll_interval: float = 5.0,
    max_wait: float = 3600.0,
) -> dict[str, Any]:
    """Poll until the build reaches a terminal state, or *max_wait* passes.

    Transient status failures are retried rather than ending the wait: a build
    that is running is not affected by this side failing to ask about it.
    """
    loop = asyncio.get_event_loop()
    deadline = loop.time() + max_wait
    while True:
        try:
            data = await build_status(platform, build_id)
            if data.get("status") in TERMINAL_STATUSES:
                return data
        except HudException:
            pass
        if loop.time() >= deadline:
            return {"status": "TIMED_OUT"}
        await asyncio.sleep(poll_interval)


async def deploy(
    context: Path,
    *,
    name: str,
    platform: PlatformClient,
    registry_id: str | None = None,
    env_vars: dict[str, str] | None = None,
    build_args: dict[str, str] | None = None,
    build_secrets: dict[str, str] | None = None,
    runtime: str | None = None,
    runtime_config: dict[str, Any] | None = None,
    no_cache: bool = False,
    max_wait: float = 3600.0,
) -> BuildOutcome:
    """Build *context* on the platform as the environment *name*, and wait.

    The whole exchange in one call, for callers that only want the result —
    ``harbor.publish()`` deploying a dataset's env groups, say. Drive
    :func:`create_build`, :func:`upload_context`, :func:`trigger_build` and
    :func:`await_build` directly to do something in between, as the CLI does
    to stream the build's logs.

    A build that runs and fails comes back as a :class:`BuildOutcome` whose
    ``status`` says so; only being unable to carry the deploy out at all
    raises.
    """
    tarball, _, _, _ = await asyncio.to_thread(create_build_context_tarball, context)
    try:
        upload_url, reserved = await create_build(platform)
        await upload_context(upload_url, tarball)
    except HudException:
        raise
    except Exception as error:
        raise DeployError(f"could not upload the build context: {error}") from error
    finally:
        tarball.unlink(missing_ok=True)

    build_id, resolved_registry = await trigger_build(
        platform,
        build_id=reserved,
        name=name,
        registry_id=registry_id,
        env_vars=env_vars,
        build_args=build_args,
        build_secrets=build_secrets,
        runtime=runtime,
        runtime_config=runtime_config,
        no_cache=no_cache,
    )
    final = await await_build(platform, build_id, max_wait=max_wait)
    return BuildOutcome(
        build_id=build_id,
        registry_id=resolved_registry,
        status=str(final.get("status", "UNKNOWN")),
        version=final.get("version"),
        image_uri=final.get("uri") or final.get("image_name"),
        registry_name=final.get("registry_name"),
        lock=final.get("lock"),
    )


__all__ = [
    "TERMINAL_STATUSES",
    "BuildOutcome",
    "DeployError",
    "await_build",
    "build_status",
    "create_build",
    "deploy",
    "trigger_build",
    "upload_context",
]
