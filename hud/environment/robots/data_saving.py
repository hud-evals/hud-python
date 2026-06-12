"""Trajectory data saving for robot envs: the framework-default recorder + the
LeRobot v3 dataset sink.

:func:`default_recorder` builds the recorder from launch-time env vars alone (the
author writes zero recorder code); ``RobotEndpoint`` calls it and ``bridge.stop()``
closes it. Config by env var so the same env module works everywhere:

- ``HUD_RECORD_DIR`` — record every tick as a LeRobot v3 dataset here.
- ``HUD_HF_REPO`` — also push the dataset to this HF namespace (``HF_TOKEN``);
  ``HUD_HF_PRIVATE=1`` makes it private.
- HUD telemetry on (``HUD_API_KEY``) — stream the same ticks to the platform.

The sink, :class:`LeRobotTraceSink`, is a :class:`~hud.telemetry.TraceSink` that
turns the recorded ``(observation, action, reward, done)`` stream into a `LeRobot v3
dataset <https://github.com/huggingface/lerobot>`_ (``data/*.parquet`` +
``videos/*.mp4`` + ``meta/*.json``). Its schema is generated from the env contract
(feature names/shapes/dtypes -> LeRobot ``features``; ``robot_type`` / ``control_rate``
-> ``robot_type`` / ``fps``), extended with the RL columns ``next.reward`` / ``next.done``.

All sink work runs on the recorder's background thread, and the heavy
LeRobot/``datasets``/``pyarrow``/``av`` imports stay deferred until a dataset is built.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from hud.telemetry.recorder import TraceSink

if TYPE_CHECKING:
    from hud.telemetry import EpisodeRecorder
    from hud.telemetry.recorder import Frame

logger = logging.getLogger(__name__)


# ── contract -> LeRobot feature schema ───────────────────────────────────────


def _names(feature: dict, base: str) -> list[str]:
    """The feature's element names, or a generated default sized to its shape."""
    names = feature.get("names")
    if names:
        return list(names)
    shape = feature.get("shape") or []
    if feature.get("dtype") == "image":
        return ["height", "width", "channel"]
    n = int(shape[0]) if len(shape) == 1 else int(np.prod(shape or [1]))
    return [f"{base}_{i}" for i in range(n)]


def contract_to_lerobot_features(
    contract: dict, *, use_videos: bool = True
) -> tuple[dict[str, dict], dict[str, str]]:
    """Build a LeRobot ``features`` dict + a wire->LeRobot key map from a contract.

    Mapping (by ``role`` / ``dtype``):

    - image observation  -> ``observation.images.<name>`` (``video`` or ``image``)
    - vector observation -> ``observation.state`` (single) or ``observation.<name>``
    - string observation -> dropped (recorded as the LeRobot ``task``, not a column)
    - action             -> ``action``

    Plus the RL columns ``next.reward`` (float32 ``[1]``) and ``next.done``
    (bool ``[1]``). Returns ``(features, key_map)`` where ``key_map`` maps each
    *observation array* wire name to its LeRobot key (the action is handled
    separately, since it is not part of the observation dict).
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
        role, dtype = f.get("role"), f.get("dtype")
        if role == "observation":
            if dtype == "image":
                key = f"observation.images.{name}"
                features[key] = {
                    "dtype": img_dtype,
                    "shape": tuple(f["shape"]),
                    "names": _names(f, name),
                }
                key_map[name] = key
            elif dtype == "string":
                continue  # language conditioning -> LeRobot "task"
            else:
                key = (
                    "observation.state"
                    if (name == "state" or single_state)
                    else f"observation.{name}"
                )
                features[key] = {
                    "dtype": dtype,
                    "shape": tuple(f["shape"]),
                    "names": _names(f, name),
                }
                key_map[name] = key
        elif role == "action":
            features["action"] = {
                "dtype": dtype,
                "shape": tuple(f["shape"]),
                "names": _names(f, "action"),
            }

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


# ── the LeRobot dataset sink ──────────────────────────────────────────────────


class LeRobotTraceSink(TraceSink):
    """Write recorded episodes into a single local LeRobot v3 dataset.

    One sink == one dataset (all episodes recorded by a serving env process). The
    dataset is created lazily on the first episode (so an env that is never driven
    leaves no artifacts), and finalized on :meth:`on_close`.
    """

    def __init__(
        self,
        contract: dict,
        root: str | Path,
        repo_id: str,
        *,
        fps: float | None = None,
        robot_type: str | None = None,
        model_contract: dict | None = None,
        use_videos: bool = True,
        push_to_hub: bool = False,
        private: bool = False,
    ) -> None:
        self._contract = contract
        self._root = Path(root)
        self._repo_id = repo_id
        #: Push the finalized dataset to the HF Hub (``repo_id`` namespace) on close.
        self._push_to_hub = push_to_hub
        self._private = private
        self._fps = round(fps if fps is not None else contract.get("control_rate", 10))
        self._robot_type = robot_type or contract.get("robot_type")
        self._model_contract = model_contract
        self._use_videos = use_videos
        self._features, self._key_map = contract_to_lerobot_features(
            contract, use_videos=use_videos
        )
        self._ds: Any | None = None
        self._task: str = ""
        self._episode_open = False
        self._episode_frames = 0

    # ── TraceSink interface (worker thread only) ──────────────────────────────

    def on_episode_start(self, meta: dict[str, Any]) -> None:
        prompt = meta.get("prompt", meta.get("task", ""))
        self._task = prompt if isinstance(prompt, str) else ""
        self._episode_open = True
        self._episode_frames = 0
        self._ensure_dataset()

    def on_frame(self, frame: Frame) -> None:
        self._ensure_dataset()
        row: dict[str, Any] = {}
        for wire, key in self._key_map.items():
            value = frame.obs.get(wire)
            if value is None:
                logger.warning("obs missing wire feature %r; skipping frame", wire)
                return
            ft = self._features[key]
            if ft["dtype"] in ("video", "image"):
                row[key] = _as_hwc_uint8(value)
            else:
                row[key] = np.asarray(value, dtype=ft["dtype"]).reshape(ft["shape"])

        act_ft = self._features["action"]
        row["action"] = np.asarray(frame.action, dtype=act_ft["dtype"]).reshape(act_ft["shape"])
        row["next.reward"] = np.asarray([frame.reward], dtype=np.float32)
        row["next.done"] = np.asarray([frame.done], dtype=bool)
        row["task"] = self._task
        self._ds.add_frame(row)
        self._episode_frames += 1

    def on_episode_end(self, meta: dict[str, Any]) -> None:
        if self._ds is None or not self._episode_open:
            return
        if self._episode_frames > 0:
            self._ds.save_episode()
        elif self._ds.has_pending_frames():
            self._ds.clear_episode_buffer()
        self._episode_open = False
        self._episode_frames = 0

    def on_close(self) -> None:
        if self._ds is None:
            return
        # Flush a trailing, never-ended episode (e.g. abrupt shutdown).
        if self._episode_open and self._episode_frames > 0:
            self._ds.save_episode()
        self._ds.finalize()
        logger.info("finalized LeRobot dataset at %s", self._root)
        if self._push_to_hub:
            self._push()

    def _push(self) -> None:
        """Push the finalized dataset to the HF Hub (best-effort; never raises).

        Uses the standard ``HF_TOKEN`` for auth. A failure (bad/missing token,
        network) is logged and swallowed — the on-disk dataset is the source of
        truth, so a push hiccup never loses data or crashes the env.
        """
        try:
            self._ds.push_to_hub(private=self._private)
            url = f"https://huggingface.co/datasets/{self._repo_id}"
            logger.info("pushed dataset to HF: %s", url)
            print(f"[env] pushed dataset -> {url}", flush=True)
        except Exception as exc:
            logger.exception("HF push failed for %s", self._repo_id)
            print(
                f"[env] WARNING: HF push failed for {self._repo_id}: {exc!r} "
                "(dataset is still on disk)",
                flush=True,
            )

    # ── internals ─────────────────────────────────────────────────────────────

    def _ensure_dataset(self) -> None:
        if self._ds is not None:
            return
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:  # missing parquet/video extras
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
        self._write_provenance()

    def _write_provenance(self) -> None:
        """Stash the raw env (+ optional model) contract for downstream tooling."""
        payload: dict[str, Any] = {"env_contract": self._contract}
        if self._model_contract is not None:
            payload["model_contract"] = self._model_contract
        meta_dir = self._root / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / "hud_contract.json").write_text(json.dumps(payload, indent=2))


# ── the framework-default recorder ────────────────────────────────────────────


def _lerobot_sink(contract: dict, record_dir: str, *, name: str):
    """Build the LeRobot dataset sink under ``<record_dir>/<name>_<stamp>/``.

    If ``HUD_HF_REPO`` (an HF namespace) is set, the dataset is also pushed to
    ``<HUD_HF_REPO>/<name>_<stamp>`` — durable even on ephemeral disk.
    """
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
    """Build the framework-default recorder from launch-time config.

    One :class:`~hud.telemetry.EpisodeRecorder` fanning out to every enabled sink
    (see the module docstring), or ``None`` if nothing is enabled.
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


__all__ = ["LeRobotTraceSink", "contract_to_lerobot_features", "default_recorder"]
