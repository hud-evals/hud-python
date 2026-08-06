"""``BashGrader`` — run a shell command and score by exit code."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from typing_extensions import override

from hud.utils.process import create_process_group_exec

from .base import Grader
from .results import SubScore

logger = logging.getLogger(__name__)


class BashGrader(Grader):
    """Run a shell command and score by exit code. Fully async."""

    name = "BashGrader"

    default_timeout: int = 600

    @classmethod
    @override
    async def compute_score(
        cls,
        command: str | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        **kwargs: Any,
    ) -> SubScore:
        """Run ``command`` via ``bash -lc`` and score by exit code."""
        if command is None:
            raise ValueError("BashGrader requires command")
        if timeout_seconds is None:
            timeout_seconds = cls.default_timeout
        del kwargs
        logger.info(
            "Running grader command: %s (cwd=%s, timeout=%ss)", command, cwd, timeout_seconds
        )
        try:
            process = await create_process_group_exec(
                "/bin/bash",
                "-lc",
                command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return SubScore(
                name=cls.name,
                value=0.0,
                info={
                    "exit_code": None,
                    "stdout": "",
                    "stderr": "/bin/bash not found",
                    "timed_out": False,
                },
            )
        result = await process.complete(max_wait=timeout_seconds)
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
                    "timeout": timeout_seconds,
                },
            )

        returncode = result.returncode if result.returncode is not None else 1
        return SubScore(
            name=cls.name,
            value=1.0 if returncode == 0 else 0.0,
            info={"exit_code": returncode, "stdout": stdout, "stderr": stderr},
        )


__all__ = ["BashGrader"]
