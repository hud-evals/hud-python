"""Safety properties of the shared Docker CLI runner.

These tests execute tiny Python programs in place of ``docker``.  They exercise
the real subprocess boundary without requiring or mutating a Docker daemon.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import sys
import time
from typing import TYPE_CHECKING, cast

import pytest

from hud.eval import runtime as runtime_module

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any


def _write_cli(tmp_path: Path, source: str) -> Path:
    script = tmp_path / "fake_docker.py"
    script.write_text(source)
    return script


@pytest.fixture
def python_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve ``docker`` to Python so the first CLI argument is a test script."""

    monkeypatch.setattr(
        runtime_module.shutil,
        "which",
        lambda executable: sys.executable if executable == "docker" else None,
    )


def _wait_for_file(path: Path, *, wait_seconds: float = 3.0) -> None:
    deadline = time.monotonic() + wait_seconds
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(0.01)


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def test_docker_cli_retains_only_bounded_output_tails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    python_docker: None,
) -> None:
    limit = 32
    stdout = "discard-stdout-" + "o" * 100 + "-stdout-tail"
    stderr = "discard-stderr-" + "e" * 100 + "-stderr-tail"
    script = _write_cli(
        tmp_path,
        f"import sys\nsys.stdout.write({stdout!r})\nsys.stderr.write({stderr!r})\n",
    )
    monkeypatch.setattr(runtime_module, "_DOCKER_OUTPUT_LIMIT", limit)

    out, err = await runtime_module._docker(str(script))

    assert out == stdout[-limit:]
    assert err == stderr[-limit:]


async def test_docker_cli_uses_devnull_stdin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    python_docker: None,
) -> None:
    script = _write_cli(
        tmp_path,
        "import sys\nassert sys.stdin.buffer.read() == b''\nprint('stdin-eof')\n",
    )
    real_create_subprocess_exec = asyncio.create_subprocess_exec
    seen: dict[str, object] = {}

    async def recording_create_subprocess_exec(
        *args: str, **kwargs: object
    ) -> asyncio.subprocess.Process:
        seen["stdin"] = kwargs.get("stdin")
        return await cast("Any", real_create_subprocess_exec)(*args, **kwargs)

    monkeypatch.setattr(
        runtime_module.asyncio,
        "create_subprocess_exec",
        recording_create_subprocess_exec,
    )

    out, _ = await runtime_module._docker(str(script))

    assert out.strip() == "stdin-eof"
    assert seen["stdin"] is asyncio.subprocess.DEVNULL


async def test_docker_cli_unsets_task_environment_before_trusted_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    python_docker: None,
) -> None:
    task_only = "HUD_TEST_DOCKER_TASK_ONLY"
    overridden = "HUD_TEST_DOCKER_OVERRIDE"
    trusted_only = "HUD_TEST_DOCKER_TRUSTED_ONLY"
    script = _write_cli(
        tmp_path,
        "import json, os, sys\n"
        "print(json.dumps({key: os.environ.get(key) for key in sys.argv[1:]}))\n",
    )
    monkeypatch.setenv(task_only, "task-value")
    monkeypatch.setenv(overridden, "task-value")
    monkeypatch.delenv(trusted_only, raising=False)

    out, _ = await runtime_module._docker(
        str(script),
        task_only,
        overridden,
        trusted_only,
        "PATH",
        env={overridden: "trusted-value", trusted_only: "trusted-value"},
        unset_env=(task_only, overridden, "PATH"),
    )

    values = json.loads(out)
    assert values == {
        task_only: None,
        overridden: "trusted-value",
        trusted_only: "trusted-value",
        "PATH": None,
    }


async def test_docker_cli_redacts_sensitive_arguments_on_failure(
    tmp_path: Path,
    python_docker: None,
) -> None:
    script = _write_cli(
        tmp_path,
        "import sys\n"
        "sys.stderr.write('controlled failure: ' + ' '.join(sys.argv[1:]))\n"
        "raise SystemExit(19)\n",
    )

    with pytest.raises(RuntimeError) as error:
        await runtime_module._docker(
            str(script),
            "--env",
            "API_TOKEN=visible-env-secret",
            "--env-file=/tmp/visible-env-file-secret",
            "-e=INLINE=visible-inline-secret",
            "--password",
            "visible-password-secret",
            "--secret=id=token,src=/tmp/visible-secret-file",
        )

    message = str(error.value)
    assert "controlled failure" in message
    assert "--env <redacted>" in message
    assert "--env-file=<redacted>" in message
    assert "-e=<redacted>" in message
    assert "--password <redacted>" in message
    assert "--secret=<redacted>" in message
    assert "visible-" not in message


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="process groups require POSIX")
async def test_docker_cli_cancellation_terminates_group_and_reaps_leader(
    tmp_path: Path,
    python_docker: None,
) -> None:
    parent_pid_path = tmp_path / "parent.pid"
    child_pid_path = tmp_path / "child.pid"
    parent_term_path = tmp_path / "parent.terminated"
    child_term_path = tmp_path / "child.terminated"
    script = _write_cli(
        tmp_path,
        """\
import os
import pathlib
import signal
import subprocess
import sys
import time

parent_pid, child_pid, parent_term, child_term = map(pathlib.Path, sys.argv[1:])

def terminate_parent(signum, frame):
    parent_term.write_text(str(signum))
    raise SystemExit(0)

signal.signal(signal.SIGTERM, terminate_parent)
parent_pid.write_text(str(os.getpid()))
child_source = '''\
import os
import pathlib
import signal
import sys
import time

pid_path, term_path = map(pathlib.Path, sys.argv[1:])

def terminate_child(signum, frame):
    term_path.write_text(str(signum))
    raise SystemExit(0)

signal.signal(signal.SIGTERM, terminate_child)
pid_path.write_text(str(os.getpid()))
while True:
    time.sleep(1)
'''
subprocess.Popen(
    [sys.executable, "-c", child_source, str(child_pid), str(child_term)],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
while True:
    time.sleep(1)
""",
    )
    process = asyncio.create_task(
        runtime_module._docker(
            str(script),
            str(parent_pid_path),
            str(child_pid_path),
            str(parent_term_path),
            str(child_term_path),
        )
    )

    try:
        await asyncio.gather(
            asyncio.to_thread(_wait_for_file, parent_pid_path),
            asyncio.to_thread(_wait_for_file, child_pid_path),
        )
        parent_pid = int(parent_pid_path.read_text())

        process.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(process, timeout=3.0)

        await asyncio.gather(
            asyncio.to_thread(_wait_for_file, parent_term_path),
            asyncio.to_thread(_wait_for_file, child_term_path),
        )
        assert not _pid_is_alive(parent_pid)
    finally:
        process.cancel()
        with contextlib.suppress(asyncio.CancelledError, TimeoutError):
            await asyncio.wait_for(process, timeout=1.0)
        for pid_path in (parent_pid_path, child_pid_path):
            if pid_path.exists():
                pid = int(pid_path.read_text())
                if _pid_is_alive(pid):
                    with contextlib.suppress(ProcessLookupError):
                        os.kill(pid, signal.SIGKILL)
