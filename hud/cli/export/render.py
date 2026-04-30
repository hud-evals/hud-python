"""Prompt rendering helpers for export.

Two layers, both async and reusable:

* ``render_prompts_via_url(url, tasks, taskset_name)`` — renders against
  an already-running MCP server. The platform pipeline reuses this
  against its running env containers; the CLI's docker mode wraps it.

* ``render_prompts_via_image(image, tasks, taskset_name, env_vars)`` —
  boots a container from the pushed image, waits for the MCP endpoint,
  calls the URL-based helper, then stops the container.

Both return ``{slug: rendered_text}``. Failures per task are logged and
that slug is omitted from the dict — callers can decide whether to fall
back to placeholder text.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
import shutil
import socket
import subprocess
import time
from typing import TYPE_CHECKING, Any

from .harbor import _dedupe_slugs, _full_scenario_name, _slugify

if TYPE_CHECKING:
    from collections.abc import Iterator

    from hud.eval.task import Task

LOGGER = logging.getLogger(__name__)


def _serialize_args(args: dict[str, Any] | None) -> dict[str, str]:
    """MCP prompts only accept string args; JSON-encode non-strings."""
    import json

    if not args:
        return {}
    return {k: v if isinstance(v, str) else json.dumps(v) for k, v in args.items()}


def _extract_text(messages: list[Any]) -> str:
    parts: list[str] = []
    for msg in messages:
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        text = getattr(content, "text", None)
        if text is None and isinstance(content, dict):
            text = content.get("text")
        if text is None and isinstance(content, str):
            text = content
        if text:
            parts.append(str(text))
    return "\n".join(parts).strip()


@contextlib.contextmanager
def _quiet_mcp_retry_logs() -> Iterator[None]:
    """Suppress hud.patches.mcp_patches retry warnings during a block.

    These fire on every transient ReadError from the streamable-http
    transport, which is normal during connection setup/teardown. They
    add no signal during export; just noise.
    """
    target = logging.getLogger("hud.patches.mcp_patches")
    previous = target.level
    target.setLevel(logging.ERROR)
    try:
        yield
    finally:
        target.setLevel(previous)


async def render_prompts_via_url(
    url: str,
    tasks: list[Task],
    taskset_name: str,
    *,
    timeout_per_task: float = 60.0,
) -> dict[str, str]:
    """Render prompts for each task against an already-running MCP server.

    Args:
        url: MCP HTTP URL (e.g. ``http://localhost:8080/mcp``).
        tasks: Tasks whose prompts to render.
        taskset_name: Used to resolve unqualified scenario names.
        timeout_per_task: Per-task timeout in seconds.

    Returns:
        ``{task_slug: rendered_prompt_text}``. Tasks whose render fails
        are omitted.
    """
    from fastmcp import Client
    from fastmcp.client.transports.http import StreamableHttpTransport

    rendered: dict[str, str] = {}
    slugs = _dedupe_slugs(tasks)
    with _quiet_mcp_retry_logs():
        transport = StreamableHttpTransport(url)
        client = Client(transport)
        await client.__aenter__()
        try:
            for task, slug in zip(tasks, slugs, strict=True):
                try:
                    full_name = _full_scenario_name(task, taskset_name)
                    args = _serialize_args(task.args)
                    result = await asyncio.wait_for(
                        client.get_prompt(full_name, args),
                        timeout=timeout_per_task,
                    )
                    text = _extract_text(getattr(result, "messages", []))
                    if text:
                        rendered[slug] = text
                    else:
                        LOGGER.warning("Empty prompt for %s", slug)
                except Exception as exc:
                    LOGGER.warning("Failed to render prompt for %s: %s", slug, exc)
        finally:
            await client.__aexit__(None, None, None)
    return rendered


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_port(host: str, port: int, timeout: float) -> bool:
    """Return True once a TCP connect to (host, port) succeeds."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


async def render_prompts_via_image(
    image: str,
    tasks: list[Task],
    taskset_name: str,
    *,
    env_vars: dict[str, str] | None = None,
    server_command: list[str] | None = None,
    container_port: int = 8765,
    platform: str | None = None,
    boot_timeout: float = 90.0,
    timeout_per_task: float = 60.0,
) -> dict[str, str]:
    """Boot ``image`` in Docker, render prompts via MCP, then stop it.

    Designed for the export CLI. Platforms with already-running env
    containers should call ``render_prompts_via_url`` directly.

    Args:
        image: Pushed image reference (e.g. ``ghcr.io/...@sha256:...``).
        tasks: Tasks whose prompts to render.
        taskset_name: Used to resolve unqualified scenario names.
        env_vars: Optional env vars passed to the container.
        server_command: Override the container's CMD. Defaults to
            ``["hud", "dev", "env:env", "--port", "<container_port>"]``,
            which forces HTTP mode regardless of how the image was
            originally built (many HUD images CMD with ``--stdio``).
        container_port: Port the MCP server listens on inside the image.
            Default 8765 matches ``hud dev``'s HTTP default.
        boot_timeout: Seconds to wait for the MCP server to be reachable.
        timeout_per_task: Per-task render timeout.

    Returns:
        ``{task_slug: rendered_prompt_text}``.
    """
    if shutil.which("docker") is None:
        raise RuntimeError("docker CLI not found on PATH; required for --render-prompts live")

    host_port = _free_port()
    name = f"hud-export-render-{_slugify(taskset_name)}-{secrets.token_hex(4)}"
    cmd: list[str] = [
        "docker",
        "run",
        "-d",
        "--rm",
        "--name",
        name,
        "-p",
        f"{host_port}:{container_port}",
    ]
    if platform:
        cmd.extend(["--platform", platform])
    for key, value in (env_vars or {}).items():
        cmd.extend(["-e", f"{key}={value}"])
    cmd.append(image)
    cmd.extend(
        server_command
        if server_command is not None
        else ["hud", "dev", "env:env", "--port", str(container_port)]
    )

    LOGGER.info("Booting image %s for prompt rendering", image)
    proc = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"docker run failed: {proc.stderr.strip() or proc.stdout.strip()}")
    container_id = proc.stdout.strip()

    try:
        url = f"http://localhost:{host_port}/mcp"
        if not await asyncio.to_thread(_wait_for_port, "localhost", host_port, boot_timeout):
            logs = await asyncio.to_thread(
                subprocess.run,
                ["docker", "logs", "--tail", "60", container_id],
                capture_output=True,
                text=True,
                timeout=10,
            )
            tail = (logs.stdout + logs.stderr).strip()
            tail_block = f"\n--- container logs (tail) ---\n{tail}" if tail else ""
            raise RuntimeError(
                f"MCP server at {url} did not become ready within {boot_timeout}s.{tail_block}"
            )
        return await render_prompts_via_url(
            url, tasks, taskset_name, timeout_per_task=timeout_per_task
        )
    finally:
        with contextlib.suppress(Exception):
            await asyncio.to_thread(
                subprocess.run,
                ["docker", "rm", "-f", container_id],
                capture_output=True,
                timeout=15,
            )
