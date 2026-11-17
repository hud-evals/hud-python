"""``BashGrader`` — run a shell command and score by exit code."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from hud.utils.process import create_process_group_exec

from .base import Grader, SubScore

logger = logging.getLogger(__name__)


class BashGrader(Grader):
    """Run a shell command and score ``1.0`` on exit code ``0``, else ``0.0``."""

    name = "BashGrader"

    default_timeout: float = 600

    @classmethod
    async def compute_score(
        cls,
        command: str | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        **unexpected: Any,
    ) -> SubScore:
        """Run ``command`` via ``bash -lc`` and score by exit code."""
        if unexpected:
            raise TypeError(f"BashGrader got unexpected arguments: {', '.join(sorted(unexpected))}")
        if command is None:
            raise ValueError("BashGrader requires command")
        timeout = cls.default_timeout if timeout_seconds is None else timeout_seconds
        logger.info("Running grader command: %s (cwd=%s, timeout=%ss)", command, cwd, timeout)
        process = await create_process_group_exec(
            "/bin/bash",
            "-lc",
            command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        result = await process.complete(max_wait=timeout)
        stdout = result.stdout.decode(errors="replace")
        stderr = result.stderr.decode(errors="replace")
        if result.timed_out:
            return SubScore(
                name=cls.name,
                value=0.0,
                info={
                    "exit_code": None,
                    "stdout": stdout,
                    "stderr": stderr,
                    "timed_out": True,
                    "timeout": timeout,
                },
            )
        return SubScore(
            name=cls.name,
            value=1.0 if result.returncode == 0 else 0.0,
            info={"exit_code": result.returncode, "stdout": stdout, "stderr": stderr},
        )
