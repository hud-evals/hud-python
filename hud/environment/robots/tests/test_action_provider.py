"""Unit tests for the env-side action providers (queue / merge / meta semantics)."""

from __future__ import annotations

import numpy as np
import pytest

from hud.environment.robots.action_provider import (
    RTCActionProvider,
    SyncActionProvider,
    make_action_provider,
)


def _chunk(n: int, dim: int = 2, start: float = 0.0) -> np.ndarray:
    """A [n, dim] chunk whose row i is filled with (start + i) — easy to identify."""
    return np.stack(
        [np.full((dim,), start + i, dtype=np.float32) for i in range(n)],
    )


def _hold() -> np.ndarray:
    return np.full((2,), -1.0, dtype=np.float32)


# ── factory ───────────────────────────────────────────────────────────────────


def test_make_action_provider_modes() -> None:
    sync = make_action_provider("sync")
    rtc = make_action_provider("rtc")
    assert isinstance(sync, SyncActionProvider)
    assert isinstance(rtc, RTCActionProvider)
    assert sync.mode == "sync"
    assert rtc.mode == "rtc"
    assert sync.uses_prefix is False
    assert rtc.uses_prefix is True
    assert sync.freeze_on_underrun is False


def test_make_action_provider_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="Unknown inference mode"):
        make_action_provider("nope")


def test_make_action_provider_drops_weight_for_non_weighted_modes() -> None:
    # `weight` is only a WeightedAsync kwarg; other providers must not choke on it.
    p = make_action_provider("rtc", weight=0.5)
    assert isinstance(p, RTCActionProvider)
    w = make_action_provider("weighted_async", weight=0.25)
    assert w._weight == 0.25


# ── sync: full-replace queue semantics ────────────────────────────────────────


def test_sync_full_replace_and_pop_in_order() -> None:
    p = make_action_provider("sync")
    chunk = _chunk(3)
    p.submit_chunk(chunk, obs_index=0)
    for i in range(3):
        a = p.next_action(_hold)
        np.testing.assert_array_equal(a, chunk[i])
    # exhausted -> HOLD
    np.testing.assert_array_equal(p.next_action(_hold), _hold())


def test_sync_resubmit_replaces_whole_queue() -> None:
    p = make_action_provider("sync")
    p.submit_chunk(_chunk(4, start=0.0), obs_index=0)
    p.next_action(_hold)  # consume one
    fresh = _chunk(3, start=100.0)
    p.submit_chunk(fresh, obs_index=1)
    # Full replace: execution restarts at fresh[0], old tail discarded.
    np.testing.assert_array_equal(p.next_action(_hold), fresh[0])
    assert p.obs_meta()["queue_remaining"] == 2


def test_bootstrap_holds_are_not_counted_as_underruns() -> None:
    p = make_action_provider("sync")
    for _ in range(3):  # before any chunk lands
        np.testing.assert_array_equal(p.next_action(_hold), _hold())
    assert p.stats()["underruns"] == 0
    assert p.stats()["ticks"] == 3  # HOLD ticks still advance the clock
    p.submit_chunk(_chunk(1), obs_index=0)
    p.next_action(_hold)
    p.next_action(_hold)  # post-chunk underrun
    assert p.stats()["underruns"] == 1


def test_sync_freeze_returns_none_on_underrun() -> None:
    p = make_action_provider("sync_freeze")
    assert p.freeze_on_underrun is True
    assert p.next_action(_hold) is None  # clock pauses: no tick, no HOLD
    assert p.stats()["ticks"] == 0
    assert p.stats()["underruns"] == 0


# ── queue_remaining / obs_meta ────────────────────────────────────────────────


def test_obs_meta_queue_remaining_and_unexecuted_chunk() -> None:
    p = make_action_provider("sync")
    meta = p.obs_meta()
    assert meta["queue_remaining"] == 0
    assert meta["unexecuted_chunk"] is None
    assert meta["active_chunk_obs_index"] == -1

    chunk = _chunk(4)
    p.submit_chunk(chunk, obs_index=0)
    p.next_action(_hold)
    meta = p.obs_meta()
    assert meta["queue_remaining"] == 3
    assert meta["active_chunk_obs_index"] == 0
    np.testing.assert_array_equal(meta["unexecuted_chunk"], chunk[1:])
    # The exposed tail is a copy — mutating it must not corrupt the queue.
    meta["unexecuted_chunk"][:] = 0.0
    np.testing.assert_array_equal(p.next_action(_hold), chunk[1])


def test_obs_meta_obs_index_tracks_ticks_including_holds() -> None:
    p = make_action_provider("sync")
    assert p.obs_meta()["obs_index"] == 0
    p.next_action(_hold)  # bootstrap HOLD tick
    p.submit_chunk(_chunk(2), obs_index=0)
    p.next_action(_hold)
    assert p.obs_meta()["obs_index"] == 2


# ── rtc: drop-d / replace semantics + delay measurement ───────────────────────


def test_rtc_drops_delay_prefix_on_merge() -> None:
    p = make_action_provider("rtc")
    p.submit_chunk(_chunk(8), obs_index=0)  # cold-start chunk
    for _ in range(3):  # consume 3 ticks
        p.next_action(_hold)

    fresh = _chunk(8, start=50.0)
    # Inferred from obs_index=0 while the env ran to tick 3 -> measured delay 3.
    measured = p.submit_chunk(fresh, obs_index=0)
    assert measured == 3
    # drop-d/replace: queue = fresh[3:]
    np.testing.assert_array_equal(p.next_action(_hold), fresh[3])
    assert p.obs_meta()["queue_remaining"] == 4


def test_rtc_delay_estimate_excludes_cold_start() -> None:
    p = make_action_provider("rtc", init_delay=1)
    p.next_action(_hold)
    p.next_action(_hold)
    # First chunk: measured delay (2) is real, but cold-start is excluded
    # from the buffer/stats so the estimate stays at init_delay.
    p.submit_chunk(_chunk(10), obs_index=0)
    assert p.obs_meta()["delay"] == 1
    assert p.stats()["mean_delay"] == 0.0

    for _ in range(4):
        p.next_action(_hold)
    measured = p.submit_chunk(_chunk(10), obs_index=2)  # tick 6 - 2 = 4
    assert measured == 4
    assert p.obs_meta()["delay"] == 4  # max over the buffer
    assert p.stats()["max_delay"] == 4
    assert p.stats()["n_inferences"] == 2


def test_rtc_delay_clamped_to_chunk_length() -> None:
    p = make_action_provider("rtc")
    p.submit_chunk(_chunk(2), obs_index=0)
    for _ in range(5):  # run far past the chunk
        p.next_action(_hold)
    measured = p.submit_chunk(_chunk(2), obs_index=0)
    assert measured == 2  # min(tick_delta, len(chunk))
    assert p.obs_meta()["queue_remaining"] == 0  # whole chunk dropped


def test_reset_clears_queue_and_counters() -> None:
    p = make_action_provider("rtc")
    p.submit_chunk(_chunk(5), obs_index=0)
    p.next_action(_hold)
    p.reset()
    meta = p.obs_meta()
    assert meta["queue_remaining"] == 0
    assert meta["obs_index"] == 0
    assert meta["active_chunk_obs_index"] == -1
    assert p.stats()["n_inferences"] == 0


# ── weighted_async: blend over the overlap ────────────────────────────────────


def test_weighted_async_blends_overlap_with_old_tail() -> None:
    p = make_action_provider("weighted_async", weight=0.7)
    old = _chunk(4, start=0.0)
    p.submit_chunk(old, obs_index=0)
    p.next_action(_hold)  # pos=1, old tail = old[1:4]

    fresh = _chunk(4, start=100.0)
    p.submit_chunk(fresh, obs_index=0)  # cold chunk already landed; delay = 1 tick
    new = fresh[1:]  # drop-d prefix
    expected_first = 0.7 * new[0] + 0.3 * old[1]
    np.testing.assert_allclose(p.next_action(_hold), expected_first, rtol=1e-6)
