"""Framework-default trajectory recording for robot envs.

One function, :func:`default_recorder`, builds the recorder an env should run
from launch-time configuration alone — the env author writes zero recorder
code. ``RobotEndpoint(bridge, contract=...)`` calls it and attaches the result
to the bridge; the recorder is closed by ``bridge.stop()`` (the env's
``@env.shutdown`` hook), which the serving entry point
(``python -m hud.environment.server``) always runs on shutdown.

Configuration is by environment variable, so the same declare-only env module
works everywhere (local child process, container CMD, remote sandbox):

- ``HUD_RECORD_DIR`` — record every executed tick as a LeRobot v3 dataset
  under this directory.
- ``HUD_HF_REPO`` — additionally push the finalized dataset to this Hugging
  Face namespace (uses the standard ``HF_TOKEN``); ``HUD_HF_PRIVATE=1`` makes
  the repo private.
- HUD telemetry configured (``HUD_API_KEY`` + telemetry enabled) — stream the
  same ticks live to the platform.

The heavy LeRobot imports stay deferred until a dataset sink is actually
built, so importing this module (or running without recording) never pulls
them in.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hud.telemetry import EpisodeRecorder


def _lerobot_sink(contract: dict, record_dir: str, *, name: str):
    """Build the file-backed LeRobot dataset sink under ``<record_dir>/<name>_<stamp>/``.

    If ``HUD_HF_REPO`` is set (a HF namespace, e.g. ``my-user`` or ``my-org``),
    the finalized dataset is pushed to ``<HUD_HF_REPO>/<name>_<stamp>`` on the
    Hub — so run data stays durable even when the env ran on ephemeral disk.
    """
    from hud.telemetry.lerobot import LeRobotTraceSink

    stamp = time.strftime("%Y%m%d_%H%M%S")
    root = Path(record_dir) / f"{name}_{stamp}"
    hf_repo = os.environ.get("HUD_HF_REPO")  # HF namespace -> enables the push
    push = bool(hf_repo)
    repo_id = f"{hf_repo}/{name}_{stamp}" if push else f"hud/{name}_{stamp}"
    private = os.environ.get("HUD_HF_PRIVATE", "0") not in ("0", "", "false", "False")
    sink = LeRobotTraceSink(
        contract, root=root, repo_id=repo_id, push_to_hub=push, private=private
    )
    dest = f" -> push to hf:{repo_id} ({'private' if private else 'public'})" if push else ""
    print(f"[env] recording traces -> {root}{dest}", flush=True)
    return sink


def default_recorder(contract: dict, *, name: str) -> EpisodeRecorder | None:
    """Build the framework-default recorder from launch-time configuration.

    One :class:`~hud.telemetry.EpisodeRecorder` fanning out to every sink the
    launch configuration enables (see the module docstring). Returns ``None``
    when nothing is enabled, so the bridge skips all recording overhead.
    Called by ``RobotEndpoint(bridge, contract=...)``; authors normally never
    call this directly.
    """
    sinks: list = []

    record_dir = os.environ.get("HUD_RECORD_DIR")
    if record_dir:
        sinks.append(_lerobot_sink(contract, record_dir, name=name))

    try:
        from hud.settings import settings

        if settings.telemetry_enabled and settings.api_key:
            from hud.telemetry.platform_sink import PlatformTraceSink

            sinks.append(PlatformTraceSink(env_name=name))
            print("[env] streaming ticks to the HUD platform", flush=True)
    except Exception:  # settings unavailable -> platform streaming off
        pass

    if not sinks:
        return None
    from hud.telemetry import EpisodeRecorder

    return EpisodeRecorder(*sinks)


__all__ = ["default_recorder"]
