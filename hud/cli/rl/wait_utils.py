from __future__ import annotations

import contextlib
import os
import select
import sys
import threading
import time as _time
from typing import TYPE_CHECKING

from watchfiles import watch

if TYPE_CHECKING:
    from pathlib import Path


def wait_for_enter_cancel_or_change(file_path: Path) -> tuple[bool, bool, bool]:
    """Block until Enter (start), 'q' (cancel), or file change.

    Returns (start_training, cancelled, changed).
    - start_training: True if Enter (or any non-'q' line on POSIX) was received
    - cancelled: True if 'q' was received or Ctrl-C
    - changed: True if the file changed on disk
    """
    start_training = False
    cancelled = False
    changed = False

    stop_evt: threading.Event = threading.Event()
    changed_evt: threading.Event = threading.Event()

    def _watcher() -> None:
        with contextlib.suppress(Exception):
            for _ in watch(file_path, stop_event=stop_evt, debounce=200):
                changed_evt.set()
                break

    t = threading.Thread(target=_watcher, daemon=True)
    t.start()

    try:
        if os.name == "nt":
            import msvcrt  # type: ignore[attr-defined]

            while True:
                if changed_evt.is_set():
                    changed = True
                    break

                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch in ("\r", "\n"):
                        start_training = True
                        break
                    if ch.lower() == "q":
                        cancelled = True
                        break
                _time.sleep(0.15)
        else:
            while True:
                if changed_evt.is_set():
                    changed = True
                    break

                rlist, _, _ = select.select([sys.stdin], [], [], 0.25)
                if rlist:
                    line = sys.stdin.readline()
                    if line is None:
                        continue
                    stripped = line.strip().lower()
                    if stripped == "q":
                        cancelled = True
                        break
                    # Any other (including empty) => start
                    start_training = True
                    break
                _time.sleep(0.05)

    except KeyboardInterrupt:
        cancelled = True
    finally:
        stop_evt.set()
        with contextlib.suppress(Exception):
            t.join(timeout=1)

    return start_training, cancelled, changed


__all__ = ["wait_for_enter_cancel_or_change"]
