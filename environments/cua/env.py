"""CUA environment - a virtual Linux desktop served over an `rfb` (VNC) capability.

The desktop (Xvfb :1 + x11vnc on VNC port 5900 + xfce4 + chromium) is launched as subprocesses
by `@env.initialize` (dropped to the unprivileged `ubuntu` user in the packaged image), which
then publishes the `rfb` screen so the harness's computer-use agent can drive it. Tasks grade
server-side via deterministic `BashGrader` checks plus an optional LLM judge.
"""

# NOTE: do NOT add `from __future__ import annotations` here - under it a typed @env.template
# param crashes the sync/deploy manifest path (TypeAdapter on a string forward-ref). Keep
# annotations as real objects.
import asyncio
import logging
import os
import pwd
import shutil
import socket

from hud import Environment
from hud.capabilities import Capability
from hud.graders import BashGrader, LLMJudgeGrader, SubScore, combine
from hud.settings import settings

logger = logging.getLogger(__name__)

env = Environment(name="cua-template")  # literal name - `hud deploy` static-parses it

_HOST = "127.0.0.1"
_VNC_PORT = 5900  # x11vnc serves the :1 display here (_start_desktop pins -rfbport 5900)


# ── rfb (VNC desktop) capability lifecycle ────────────────────────────────────


async def _listening(host: str, port: int, timeout: float = 30.0) -> None:
    """Block until host:port accepts a connection (chromium + xfce boot is heavy)."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            socket.create_connection((host, port), timeout=0.5).close()
            return
        except OSError:
            await asyncio.sleep(0.2)
    raise RuntimeError(f"VNC server never came up on {host}:{port}")


_DESKTOP_USER = "ubuntu"
_desktop_procs: list[asyncio.subprocess.Process] = []


def _drop_to_ubuntu() -> bool:
    """Run the desktop as the unprivileged ``ubuntu`` user when we can: i.e. we're root (the
    packaged image) and that user exists. Locally (a non-root dev box) we run as ourselves."""
    if os.geteuid() != 0:
        return False
    try:
        pwd.getpwnam(_DESKTOP_USER)
    except KeyError:
        return False
    return True


async def _start_desktop() -> None:
    """Launch the virtual desktop as subprocesses - this replaces the dinit init system, so the
    one code path serves the env locally (`hud eval`) and in the packaged image.

    In the image we run as root and drop each desktop process to ``ubuntu`` (uid 1000): the
    agent's on-screen terminal is uid 1000 and cannot read the chmod-700 grading code, while the
    control-channel server stays root and can - the uid-wall, with no init system. Needs
    Xvfb/x11vnc/xfce4/chromium on PATH (Linux; macOS has no X server, so use the container).
    """
    display = ":1"
    drop = _drop_to_ubuntu()
    home = "/home/ubuntu" if drop else os.environ.get("HOME", "/root")

    async def _spawn(*cmd: str) -> asyncio.subprocess.Process:
        argv = ["sudo", "-u", _DESKTOP_USER, "env", f"HOME={home}", f"DISPLAY={display}", *cmd] if drop else list(cmd)
        return await asyncio.create_subprocess_exec(
            *argv,
            env={**os.environ, "DISPLAY": display, "HOME": home},
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

    _desktop_procs.append(await _spawn("Xvfb", display, "-screen", "0", "1280x800x24"))
    await asyncio.sleep(0.8)  # let the X server come up before clients attach
    _desktop_procs.append(await _spawn("xfce4-session"))
    _desktop_procs.append(
        await _spawn(
            "chromium",
            "--no-sandbox",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--no-first-run",
            "--start-maximized",
            "about:blank",
        )
    )
    _desktop_procs.append(
        await _spawn(
            "x11vnc",
            "-display",
            display,
            "-rfbport",
            str(_VNC_PORT),
            "-forever",
            "-shared",
            "-nopw",
            "-localhost",
        )
    )
    if shutil.which("websockify"):  # optional noVNC debug viewer on :6080
        _desktop_procs.append(await _spawn("websockify", "--web", "/usr/share/novnc", "6080", f"localhost:{_VNC_PORT}"))


@env.initialize
async def _up() -> None:
    # The desktop is launched here as subprocesses (no init system) - both in the packaged image
    # and for a local `hud eval` subprocess. The port check keeps a re-init idempotent (skip if a
    # screen already serves). Block until VNC accepts before publishing the capability.
    try:
        socket.create_connection((_HOST, _VNC_PORT), timeout=0.5).close()
    except OSError:
        logger.info("launching desktop subprocesses")
        await _start_desktop()
    await _listening(_HOST, _VNC_PORT)
    # display 0 -> VNC port 5900 + 0 (display is the VNC port offset, NOT the X display :1).
    env.add_capability(Capability.rfb(name="screen", url=f"rfb://{_HOST}", display=0))


@env.shutdown
async def _down() -> None:
    # We launched the desktop in @env.initialize, so tear it down (sudo forwards SIGTERM to the
    # dropped child). The container dies with it anyway; this matters for local subprocess runs.
    logger.info("cua-template shutting down")
    for proc in reversed(_desktop_procs):
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
    _desktop_procs.clear()


# ── task ──────────────────────────────────────────────────────────────────────


def make_prompt(description: str) -> str:
    """Format a task description into an agent prompt."""
    return f"Use computer use tools to complete the following task:\n\n{description}"


@env.template()
async def cua_task(
    prompt: str,
    bash_checks: list[dict] | None = None,
    grading_criteria: list[str] | None = None,
):
    """General CUA task: present the prompt, then grade with any combination of deterministic
    bash checks (run server-side in this container) and an LLM rubric judge.

    `combine` normalizes the positive weights to sum to 1.0, so weights are relative.

    Args:
        prompt: The task instruction shown to the agent.
        bash_checks: Optional list of {"name", "command", "weight"} for shell-based grading.
        grading_criteria: Optional rubric strings for the LLM judge (needs HUD_API_KEY).
    """
    answer = yield make_prompt(prompt)

    bash_total = sum(c.get("weight", 1.0) for c in (bash_checks or []))
    if bash_checks and bash_total <= 0:
        raise ValueError("bash check weights must sum to a positive value")
    bash_share = 0.5 if grading_criteria else 1.0

    graders: list = []

    for check in bash_checks or []:
        graders.append(
            BashGrader.grade(
                weight=check.get("weight", 1.0) / bash_total * bash_share,
                name=check["name"],
                command=check["command"],
            )
        )

    if grading_criteria:
        judge_weight = 0.5 if bash_checks else 1.0
        if settings.api_key:
            graders.append(
                LLMJudgeGrader.grade(
                    weight=judge_weight,
                    name="llm_judge",
                    answer=str(answer),
                    question=prompt,
                    criteria=[(c, 1.0) for c in grading_criteria],
                )
            )
        else:
            # No key (e.g. a keyless deploy): the judge can't run, so it scores 0 at its real
            # weight instead of erroring the trace. Never re-weight the bash checks.
            logger.warning("No HUD_API_KEY: LLM judge skipped; it scores 0 at its weight.")
            graders.append(SubScore(name="llm_judge", weight=judge_weight, value=0.0))

    if not graders:
        graders.append(SubScore(name="desktop_running", value=1.0, weight=1.0))

    yield await combine(*graders)
