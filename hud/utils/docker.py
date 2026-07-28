"""The docker CLI, invoked the same way wherever the SDK shells out to it."""

from __future__ import annotations

import asyncio

from hud.utils.process import create_process_group_exec


async def docker(*args: str, check: bool = True, deadline: float | None = None) -> tuple[str, str]:
    """Run a docker command and return decoded ``(stdout, stderr)``.

    *deadline* bounds it and tears down the process group: a build or pull
    that hangs must not hang its caller, and killing only the CLI leaves its
    children behind.
    """
    group = await create_process_group_exec(
        "docker",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    process = group.process
    try:
        out, err = await asyncio.wait_for(process.communicate(), timeout=deadline)
    except TimeoutError:
        await group.terminate()
        raise RuntimeError(f"docker {' '.join(args)} timed out after {deadline:.0f}s") from None
    if check and process.returncode != 0:
        detail = err.decode("utf-8", "replace").strip() or out.decode("utf-8", "replace").strip()
        raise RuntimeError(f"docker {' '.join(args)} failed ({process.returncode}): {detail}")
    return out.decode("utf-8", "replace"), err.decode("utf-8", "replace")
