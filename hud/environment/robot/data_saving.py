"""Off-loop trajectory recording: save the bridge's tick stream as a LeRobot v3 dataset.

The bridge produces ``(obs, action, reward, done)`` at the control rate, and recording
must never slow that loop down: :class:`LeRobotRecorder` only copies + enqueues on the
control thread; its single daemon worker does all dataset work (image/video encoding,
parquet writes) off the loop. Heavy imports (lerobot / datasets / pyarrow / av) stay
deferred until a dataset is actually built.

:meth:`LeRobotRecorder.from_env` wires this from launch-time env vars alone
(``RobotEndpoint`` builds it, ``bridge.stop()`` closes it — zero recorder code):

- ``HUD_RECORD_DIR`` — record every tick as a LeRobot v3 dataset here.
- ``HUD_HF_REPO`` — also push the dataset to this HF namespace (``HF_TOKEN``);
  ``HUD_HF_PRIVATE=1`` makes it private.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import queue
import signal
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Shutdown signals are blocked on the worker thread so the OS delivers them to the
# main thread (the only place Python runs handlers); the owning app routes them to
# ``close()``.
_SHUTDOWN_SIGNALS = frozenset(
    s for s in (getattr(signal, n, None) for n in ("SIGINT", "SIGTERM", "SIGHUP")) if s
)


def _names(feature: dict, base: str) -> list[str]:
    """The feature's element names, or a generated default sized to its shape."""
    names = feature.get("names")
    if names:
        return list(names)
    if feature.get("dtype") == "image":
        return ["height", "width", "channel"]
    shape = feature.get("shape") or []
    n = int(shape[0]) if len(shape) == 1 else int(np.prod(shape or [1]))
    return [f"{base}_{i}" for i in range(n)]


def contract_to_lerobot_features(
    contract: dict, *, use_videos: bool = True
) -> tuple[dict[str, dict], dict[str, str]]:
    """Build a LeRobot ``features`` dict + a wire->LeRobot key map from a contract.

    Image obs -> ``observation.images.<name>``; vector obs -> ``observation.state``
    (single) or ``observation.<name>``; string obs -> dropped (becomes the LeRobot
    ``task``); action -> ``action``; plus RL columns ``next.reward`` / ``next.done``.
    """
    feats = contract.get("features", {})
    vector_obs = [
        n
        for n, f in feats.items()
        if f.get("role") == "observation" and f.get("dtype") not in ("image", "string")
    ]
    single_state = len(vector_obs) == 1

    features: dict[str, dict] = {}
    key_map: dict[str, str] = {}
    img_dtype = "video" if use_videos else "image"

    for name, f in feats.items():
        role, dtype, shape = f.get("role"), f.get("dtype"), tuple(f.get("shape") or ())
        if role == "observation" and dtype != "string":  # string -> LeRobot "task"
            if dtype == "image":
                key, dtype = f"observation.images.{name}", img_dtype
            elif name == "state" or single_state:
                key = "observation.state"
            else:
                key = f"observation.{name}"
            features[key] = {"dtype": dtype, "shape": shape, "names": _names(f, name)}
            key_map[name] = key
        elif role == "action":
            features["action"] = {"dtype": dtype, "shape": shape, "names": _names(f, "action")}

    features["next.reward"] = {"dtype": "float32", "shape": (1,), "names": ["reward"]}
    features["next.done"] = {"dtype": "bool", "shape": (1,), "names": ["done"]}
    return features, key_map


def _as_hwc_uint8(value: Any) -> np.ndarray:
    """Coerce an image to a contiguous ``uint8`` array (LeRobot accepts HWC/CHW)."""
    arr = np.asarray(value)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            scaled = arr * 255.0 if float(arr.max(initial=0.0)) <= 1.0 else arr
            arr = np.clip(scaled, 0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return np.ascontiguousarray(arr)


class LeRobotRecorder:
    """Record episodes into one local LeRobot v3 dataset, off the control loop.

    :meth:`start_episode` / :meth:`record_frame` / :meth:`end_episode` only copy +
    enqueue; a daemon worker thread writes the dataset — created lazily on the first
    episode, finalized by :meth:`close` (also registered with ``atexit``: the parquet
    footer is what makes it readable), and optionally pushed to the HF Hub.
    """

    def __init__(
        self,
        contract: dict,
        root: str | Path,
        repo_id: str,
        *,
        use_videos: bool = True,
        push_to_hub: bool = False,
        private: bool = False,
    ) -> None:
        self._contract = contract
        self._root = Path(root)
        self._repo_id = repo_id
        self._push_to_hub = push_to_hub
        self._private = private
        self._fps = round(contract.get("control_rate", 10))
        self._robot_type = contract.get("robot_type")
        self._use_videos = use_videos
        self._features, self._key_map = contract_to_lerobot_features(
            contract, use_videos=use_videos
        )
        # Worker-thread-only state (dataset + current-episode bookkeeping).
        self._ds: Any | None = None
        self._task = ""
        self._episode_open = False
        self._episode_frames = 0
        self._queue: queue.Queue[tuple[str, Any] | None] = queue.Queue()
        self._closed = False
        self._worker = threading.Thread(target=self._run, name="lerobot-recorder", daemon=True)
        self._worker.start()
        atexit.register(self.close)

    @classmethod
    def from_env(cls, contract: dict, *, name: str) -> LeRobotRecorder | None:
        """Build from ``HUD_RECORD_DIR`` / ``HUD_HF_REPO`` / ``HUD_HF_PRIVATE``;
        ``None`` if recording is off."""
        record_dir = os.environ.get("HUD_RECORD_DIR")
        if not record_dir:
            return None
        stamp = time.strftime("%Y%m%d_%H%M%S")
        root = Path(record_dir) / f"{name}_{stamp}"
        hf_repo = os.environ.get("HUD_HF_REPO")  # HF namespace -> enables the push
        repo_id = f"{hf_repo or 'hud'}/{name}_{stamp}"
        private = os.environ.get("HUD_HF_PRIVATE", "0") not in ("0", "", "false", "False")
        dest = (
            f" -> push to hf:{repo_id} ({'private' if private else 'public'})" if hf_repo else ""
        )
        print(f"[env] recording traces -> {root}{dest}", flush=True)
        return cls(
            contract, root=root, repo_id=repo_id, push_to_hub=bool(hf_repo), private=private
        )

    # ── control-thread API: copy + enqueue only, never encode ────────────────

    def start_episode(self, **meta: Any) -> None:
        """Open a new episode (``meta`` carries e.g. ``prompt`` / task args)."""
        self._put(("start", dict(meta)))

    def record_frame(
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        done: bool,
        info: dict[str, Any] | None = None,  # accepted for bridge compat; not stored
    ) -> None:
        """Copy + enqueue one tick; returns immediately."""
        # Copy now so later in-place sim mutation can't corrupt a buffered frame.
        obs_copy = {k: np.array(v, copy=True) for k, v in obs.items()}
        self._put(("frame", (obs_copy, np.array(action, copy=True), float(reward), bool(done))))

    def end_episode(self, **meta: Any) -> None:
        """Close the current episode (``meta`` carries e.g. ``success`` / reward)."""
        self._put(("end", dict(meta)))

    def close(self) -> None:
        """Drain the queue, finalize the dataset, join the worker. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)  # poison pill
        self._worker.join()

    def _put(self, event: tuple[str, Any]) -> None:
        if self._closed:
            logger.warning("LeRobotRecorder is closed; dropping %s event", event[0])
            return
        self._queue.put(event)

    # ── worker thread: all dataset work ───────────────────────────────────────

    def _run(self) -> None:
        # Block shutdown signals on this thread so they always reach the main thread —
        # a signal delivered here would never run its handler, and finalize would be
        # skipped. Unix-only; must run on this thread.
        if hasattr(signal, "pthread_sigmask") and _SHUTDOWN_SIGNALS:
            with contextlib.suppress(ValueError, OSError):
                signal.pthread_sigmask(signal.SIG_BLOCK, _SHUTDOWN_SIGNALS)
        while (event := self._queue.get()) is not None:
            kind, payload = event
            try:  # one bad event must not kill the worker loop
                if kind == "start":
                    prompt = payload.get("prompt", payload.get("task", ""))
                    self._task = prompt if isinstance(prompt, str) else ""
                    self._episode_open, self._episode_frames = True, 0
                    self._ensure_dataset()
                elif kind == "frame":
                    self._write_frame(*payload)
                elif self._ds is not None and self._episode_open:  # "end"
                    if self._episode_frames > 0:
                        self._ds.save_episode()
                    elif self._ds.has_pending_frames():
                        self._ds.clear_episode_buffer()
                    self._episode_open = False
                    self._episode_frames = 0
            except Exception:
                logger.exception("recorder failed handling %s event", kind)
        try:
            self._finalize()
        except Exception:
            logger.exception("recorder failed to finalize dataset")

    def _write_frame(self, obs: dict, action: np.ndarray, reward: float, done: bool) -> None:
        self._ensure_dataset()
        row: dict[str, Any] = {}
        for wire, key in self._key_map.items():
            value = obs.get(wire)
            if value is None:
                logger.warning("obs missing wire feature %r; skipping frame", wire)
                return
            ft = self._features[key]
            if ft["dtype"] in ("video", "image"):
                row[key] = _as_hwc_uint8(value)
            else:
                row[key] = np.asarray(value, dtype=ft["dtype"]).reshape(ft["shape"])
        act_ft = self._features["action"]
        row["action"] = np.asarray(action, dtype=act_ft["dtype"]).reshape(act_ft["shape"])
        row["next.reward"] = np.asarray([reward], dtype=np.float32)
        row["next.done"] = np.asarray([done], dtype=bool)
        row["task"] = self._task
        self._ds.add_frame(row)
        self._episode_frames += 1

    def _finalize(self) -> None:
        if self._ds is None:
            return
        # Flush a trailing, never-ended episode (e.g. abrupt shutdown).
        if self._episode_open and self._episode_frames > 0:
            self._ds.save_episode()
        self._ds.finalize()
        logger.info("finalized LeRobot dataset at %s", self._root)
        if not self._push_to_hub:
            return
        try:  # best-effort: the on-disk dataset is the source of truth
            self._ds.push_to_hub(private=self._private)
            url = f"https://huggingface.co/datasets/{self._repo_id}"
            print(f"[env] pushed dataset -> {url}", flush=True)
        except Exception as exc:
            logger.exception("HF push failed for %s", self._repo_id)
            print(f"[env] WARNING: HF push failed: {exc!r} (dataset is still on disk)", flush=True)

    def _ensure_dataset(self) -> None:
        if self._ds is not None:
            return
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError(
                "Trace recording needs the LeRobot dataset extras. Install with:\n"
                "    pip install 'lerobot[dataset]' av"
            ) from exc

        # LeRobotDataset.create requires the root not to pre-exist.
        self._ds = LeRobotDataset.create(
            repo_id=self._repo_id,
            fps=self._fps,
            features=self._features,
            root=self._root,
            robot_type=self._robot_type,
            use_videos=self._use_videos,
        )
        # Stash the raw env contract for downstream tooling.
        meta_dir = self._root / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / "hud_contract.json").write_text(
            json.dumps({"env_contract": self._contract}, indent=2)
        )


__all__ = ["LeRobotRecorder", "contract_to_lerobot_features"]
