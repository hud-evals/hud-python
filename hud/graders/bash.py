"""``BashGrader`` — run a shell command and score by exit code."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from hud.utils.process import create_process_group_exec

from .base import Grader
from .results import SubScore

logger = logging.getLogger(__name__)


class BashGrader(Grader):
    """Run a shell command and score by exit code. Fully async."""

    name = "BashGrader"

    default_timeout: int = 600

    @classmethod
    async def compute_score(
        cls,
        command: str,
        cwd: str | None = None,
        timeout_seconds: int | None = None,
        **kwargs: Any,
    ) -> SubScore:
        """Run ``command`` via ``bash -lc`` and score by exit code."""
        if timeout_seconds is None:
            timeout_seconds = cls.default_timeout
        del kwargs
        logger.info(
            "Running grader command: %s (cwd=%s, timeout=%ss)", command, cwd, timeout_seconds
        )
        try:
            proc = await create_process_group_exec(
                "/bin/bash",
                "-lc",
                command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await proc.communicate(max_wait=timeout_seconds)
            stdout = stdout_bytes.decode(errors="replace")
            stderr = stderr_bytes.decode(errors="replace")
            returncode = proc.returncode if proc.returncode is not None else 1
        except TimeoutError:
            return SubScore(
                name=cls.name,
                value=0.0,
                metadata={
                    "exit_code": None,
                    "stdout": "",
                    "stderr": "",
                    "timed_out": True,
                    "timeout": timeout_seconds,
                },
            )
        except FileNotFoundError:
            return SubScore(
                name=cls.name,
                value=0.0,
                metadata={
                    "exit_code": None,
                    "stdout": "",
                    "stderr": "/bin/bash not found",
                    "timed_out": False,
                },
            )

        return SubScore(
            name=cls.name,
            value=1.0 if returncode == 0 else 0.0,
            metadata={"exit_code": returncode, "stdout": stdout, "stderr": stderr},
        )


__all__ = ["BashGrader"]
