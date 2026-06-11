"""Shared glue for adding LeRobot trace recording to an env server.

Three small helpers so every env wires recording the *same* way, instead of each
``env_server.py`` carrying its own bespoke copy:

- :func:`add_record_arg` — the uniform ``--record [DIR]`` CLI flag.
- :func:`make_recorder` — build an :class:`~hud.telemetry.EpisodeRecorder` that
  writes a LeRobot v3 dataset under ``<DIR>/<name>_<stamp>/`` (or ``None`` when
  recording is off).
- :func:`serve_until_signal` — serve the env until it returns *or* a shutdown
  signal arrives, so the caller's ``finally`` (``recorder.close()`` → dataset
  ``finalize``) always runs and the dataset on disk stays loadable.

Adding recording to a new env is then: ``add_record_arg(parser, ...)`` →
``make_recorder(contract, args.record, name=...)`` → pass ``recorder=`` to the
bridge → ``recorder.start_episode`` / ``recorder.end_episode`` per episode →
serve via :func:`serve_until_signal` with ``recorder.close()`` in ``finally``.

The heavy LeRobot imports stay deferred to :func:`make_recorder`, so importing
this module (or running without ``--record``) never pulls them in.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse

    from hud.environment import Environment
    from hud.telemetry import EpisodeRecorder


def add_record_arg(parser: argparse.ArgumentParser, *, default_dir: str | Path) -> None:
    """Add the uniform ``--record [DIR]`` flag (defaults to ``default_dir`` if bare)."""
    parser.add_argument(
        "--record",
        nargs="?",
        const=str(default_dir),
        default=None,
        help="record episodes as a LeRobot v3 dataset (optionally pass an output dir)",
    )


def make_recorder(
    contract: dict, record_dir: str | None, *, name: str
) -> EpisodeRecorder | None:
    """Build an off-loop recorder writing a LeRobot v3 dataset, or ``None`` if off.

    The dataset lands at ``<record_dir>/<name>_<stamp>/`` with metadata derived
    from ``contract``. Returns ``None`` when ``record_dir`` is ``None`` so the
    bridge skips all recording overhead.

    **Optional Hugging Face push.** If ``BENCH_HF_REPO`` is set (the user's HF
    namespace, e.g. ``my-user`` or ``my-org``), the finalized dataset is pushed to
    ``<BENCH_HF_REPO>/<name>_<stamp>`` on the Hub using the standard ``HF_TOKEN``.
    This makes the run data durable regardless of where the env ran (so cloud env
    containers, whose disk is ephemeral, still produce a persistent artifact).
    ``BENCH_HF_PRIVATE=1`` makes the repo private (default: public).
    """
    if record_dir is None:
        return None
    from hud.telemetry import EpisodeRecorder

    return EpisodeRecorder(_lerobot_sink(contract, record_dir, name=name))


def _lerobot_sink(contract: dict, record_dir: str, *, name: str):
    """Build the file-backed LeRobot dataset sink under ``<record_dir>/<name>_<stamp>/``.

    See :func:`make_recorder` for the ``BENCH_HF_REPO`` / ``BENCH_HF_PRIVATE``
    Hugging Face push behavior (it applies here — the sink owns the push).
    """
    from hud.telemetry.lerobot import LeRobotTraceSink

    stamp = time.strftime("%Y%m%d_%H%M%S")
    root = Path(record_dir) / f"{name}_{stamp}"
    hf_repo = os.environ.get("BENCH_HF_REPO")  # HF namespace -> enables the push
    push = bool(hf_repo)
    repo_id = f"{hf_repo}/{name}_{stamp}" if push else f"hud/{name}_{stamp}"
    private = os.environ.get("BENCH_HF_PRIVATE", "0") not in ("0", "", "false", "False")
    sink = LeRobotTraceSink(
        contract, root=root, repo_id=repo_id, push_to_hub=push, private=private
    )
    dest = f" -> push to hf:{repo_id} ({'private' if private else 'public'})" if push else ""
    print(f"[env] recording traces -> {root}{dest}", flush=True)
    return sink


def default_recorder(contract: dict, *, name: str) -> EpisodeRecorder | None:
    """Build the framework-default recorder from launch-time configuration.

    One :class:`~hud.telemetry.EpisodeRecorder` fanning out to every sink the
    launch configuration enables — the env author writes no recorder code:

    - **LeRobot dataset** (``BENCH_RECORD_DIR`` set): every executed tick lands
      in a LeRobot v3 dataset under that directory (per-lane dirs come from the
      fleet; the optional HF push applies, see :func:`make_recorder`).
    - **Platform stream** (HUD telemetry configured: ``HUD_API_KEY`` set and
      telemetry enabled): the same tick stream ships live to the platform via
      :class:`~hud.telemetry.platform_sink.PlatformTraceSink`.

    Returns ``None`` when nothing is enabled, so the bridge skips all recording
    overhead. Called by ``RobotEndpoint(bridge, contract=...)``; authors normally
    never call this directly.
    """
    sinks: list = []

    record_dir = os.environ.get("BENCH_RECORD_DIR")
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


async def serve_until_signal(env: Environment, host: str, port: int) -> None:
    """Run ``env.serve(host, port)`` until it returns or a shutdown signal arrives.

    Returns on ``SIGTERM`` (``kill``) / ``SIGHUP`` (closed terminal) so the
    caller's ``finally`` runs and a recorder can finalize a loadable dataset.
    ``SIGINT`` (Ctrl-C) already surfaces as ``KeyboardInterrupt`` through the
    caller. ``add_signal_handler`` is the reliable path for an asyncio app that
    also runs the recorder's background thread.
    """
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGHUP):
        # Suppressed: signals unavailable (non-Unix) or loop not on the main thread;
        # rely on KeyboardInterrupt / the caller's own shutdown path instead.
        with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
            loop.add_signal_handler(sig, stop.set)

    serve_task = asyncio.ensure_future(env.serve(host, port))
    stop_task = asyncio.ensure_future(stop.wait())
    try:
        done, _ = await asyncio.wait(
            {serve_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if serve_task in done:
            serve_task.result()  # surface a server error if serve() returned
    finally:
        for task in (serve_task, stop_task):
            task.cancel()
        with contextlib.suppress(Exception):
            await asyncio.gather(serve_task, stop_task, return_exceptions=True)


__all__ = ["add_record_arg", "default_recorder", "make_recorder", "serve_until_signal"]
