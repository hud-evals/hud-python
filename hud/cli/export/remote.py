"""Remote-mode export client.

When ``hud export <fmt> --taskset NAME`` is invoked, the platform owns
the entire export pipeline (Harbor build, image mirroring, S3 upload).
This module just resolves the name → evalset_id, kicks off the job,
polls until done, and extracts the resulting tarball.

Required because users behind private ECR can't pull env images
locally — all docker pull/push must happen server-side.
"""

from __future__ import annotations

import logging
import tarfile
import tempfile
import time
from typing import TYPE_CHECKING, Any

import httpx
import typer

from hud.cli.utils.api import hud_headers, require_api_key
from hud.cli.utils.taskset import resolve_taskset_id
from hud.settings import settings
from hud.utils.hud_console import HUDConsole

if TYPE_CHECKING:
    from pathlib import Path

LOGGER = logging.getLogger(__name__)

_TERMINAL_STATES = {"completed", "error"}


def remote_export(
    *,
    taskset_name: str,
    output_dir: Path,
    fmt: str = "harbor",
    private: bool = False,
    poll_interval: float = 3.0,
    timeout: float = 3600.0,
    console: HUDConsole | None = None,
) -> Path:
    """Run an export on the platform and extract the result locally.

    Args:
        taskset_name: Name (or UUID) of the taskset on the platform.
        output_dir: Directory to extract the resulting tarball into.
        fmt: Export format (e.g. ``harbor``).
        private: Push mirrored images to private Docker Hub repos.
        poll_interval: Seconds between status polls.
        timeout: Max seconds to wait for the export to complete.

    Returns:
        ``output_dir`` (resolved, absolute).

    Raises ``typer.Exit(1)`` on any failure with a user-facing message.
    """
    hud_console = console or HUDConsole()
    require_api_key("export a taskset")

    api_url = settings.hud_api_url.rstrip("/")
    headers = hud_headers()

    hud_console.progress_message(f"Resolving taskset '{taskset_name}'...")
    evalset_id, _, _ = resolve_taskset_id(taskset_name, api_url, headers, create=False)
    if not evalset_id:
        hud_console.error(f"Taskset '{taskset_name}' not found on platform")
        raise typer.Exit(1)

    hud_console.progress_message(f"Requesting {fmt} export...")
    try:
        response = httpx.post(
            f"{api_url}/exports/tasksets/{evalset_id}",
            json={"format": fmt, "private": private},
            headers=headers,
            timeout=60.0,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        _surface_http_error(exc, hud_console)
        raise typer.Exit(1) from None
    export_id = response.json()["export_id"]
    hud_console.success(f"Export queued: {export_id}")

    row = _poll_until_terminal(
        api_url=api_url,
        headers=headers,
        export_id=export_id,
        poll_interval=poll_interval,
        timeout=timeout,
        console=hud_console,
    )
    if row.get("status") == "error":
        hud_console.error(f"Export failed: {row.get('error') or 'unknown error'}")
        raise typer.Exit(1)

    hud_console.progress_message("Fetching download URL...")
    try:
        dl_response = httpx.get(
            f"{api_url}/exports/{export_id}/download",
            headers=headers,
            timeout=30.0,
        )
        dl_response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        _surface_http_error(exc, hud_console)
        raise typer.Exit(1) from None
    presigned_url = dl_response.json()["url"]

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    hud_console.progress_message(f"Downloading and extracting to {output_dir}...")
    _download_and_extract(presigned_url, output_dir)

    hud_console.success(f"Extracted to {output_dir}")
    return output_dir


def _poll_until_terminal(
    *,
    api_url: str,
    headers: dict[str, str],
    export_id: str,
    poll_interval: float,
    timeout: float,
    console: HUDConsole,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_status = ""
    while True:
        if time.monotonic() > deadline:
            console.error(f"Export {export_id} timed out after {timeout:.0f}s")
            raise typer.Exit(1)
        try:
            poll_response = httpx.get(
                f"{api_url}/exports/{export_id}",
                headers=headers,
                timeout=30.0,
            )
            poll_response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            _surface_http_error(exc, console)
            raise typer.Exit(1) from None
        row = poll_response.json()
        status = row.get("status", "")
        if status != last_status:
            console.progress_message(f"Status: {status}")
            last_status = status
        if status in _TERMINAL_STATES:
            return row
        time.sleep(poll_interval)


def _download_and_extract(url: str, output_dir: Path) -> None:
    """Stream the presigned URL to a temp tarball, then extract."""
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=True) as tmp:
        with httpx.stream("GET", url, timeout=300.0) as stream:
            stream.raise_for_status()
            for chunk in stream.iter_bytes():
                tmp.write(chunk)
        tmp.flush()
        with tarfile.open(tmp.name, "r:gz") as tar:
            tar.extractall(output_dir, filter="data")


def _surface_http_error(exc: httpx.HTTPStatusError, console: HUDConsole) -> None:
    """Translate an HTTP error from the platform into a user-facing message."""
    code = exc.response.status_code
    try:
        body = exc.response.json()
    except Exception:
        body = exc.response.text

    detail: Any = body.get("detail") if isinstance(body, dict) else body

    if code == 401:
        console.error("Authentication failed. Run `hud login` or set HUD_API_KEY.")
    elif code == 403:
        console.error(f"Forbidden: {detail}")
    elif code == 404:
        console.error(f"Not found: {detail}")
    elif code == 409:
        if isinstance(detail, dict) and detail.get("message"):
            console.error(f"Export precheck failed: {detail['message']}")
        else:
            console.error(f"Conflict: {detail}")
    else:
        console.error(f"HTTP {code}: {detail}")
