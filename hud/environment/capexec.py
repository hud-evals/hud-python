"""Execute a trusted process without inherited ambient capabilities."""

from __future__ import annotations

import ctypes
import os
import sys
from typing import NoReturn

_PR_CAP_AMBIENT = 47
_PR_CAP_AMBIENT_CLEAR_ALL = 4


def exec_without_ambient_capabilities(argv: list[str]) -> NoReturn:
    if not argv:
        raise ValueError("command required")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_CAP_AMBIENT, _PR_CAP_AMBIENT_CLEAR_ALL, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    os.execvp(argv[0], argv)  # noqa: S606 - replace this trusted trampoline process


if __name__ == "__main__":
    exec_without_ambient_capabilities(sys.argv[1:])
