"""Local Docker-backed runtime for Harbor task directories."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import tempfile
import tomllib
import uuid
from collections.abc import AsyncGenerator  # noqa: TC003 - env.template resolves this at runtime.
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hud.environment import Environment
from hud.environment.workspace import Workspace
from integrations.harbor_common import _hash_directory, _slugify, _task_dirs

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import asyncssh

    from hud.eval import Task
    from hud.eval.runtime import Runtime


class HarborRuntime:
    """Run Harbor task directories through HUD's local rollout engine.

    The provider builds the Harbor task's ``environment/`` Docker context, then
    materializes the built image's working directory onto a writable host
    workspace and bind-mounts it back over the same guest path. Because the
    workspace is the image's actual working directory (source *plus* every file
    the build generated — start scripts, installed dependencies, compiled output,
    seeded databases — with their original mode bits), the agent sees exactly
    what the image would run, and edits made over SFTP are visible to the running
    process. If the task ships a ``docker-compose.yaml``/``.yml``, the provider
    starts it with an overlay that keeps the ``main`` service idle while
    preserving sidecars such as databases. Shell commands execute inside the main
    container via ``docker exec``. Grading runs the Harbor ``tests/test.sh``
    inside the same main container, bounded by the task's ``[verifier]
    timeout_sec``, and reads ``/logs/verifier/reward.json`` or ``reward.txt``.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        ready_timeout: float = 120.0,
    ) -> None:
        self.root = Path(path).resolve()
        self.ready_timeout = ready_timeout
        self._task_dirs = {task_dir.name: task_dir for task_dir in _task_dirs(self.root)}
        if not self._task_dirs:
            raise ValueError(f"no Harbor tasks found in {path}")

    @contextlib.asynccontextmanager
    async def __call__(self, task: Task) -> AsyncIterator[Runtime]:
        from hud.eval.runtime import Runtime, _docker, _local

        task_dir = self._task_dirs.get(task.id)
        if task_dir is None:
            raise KeyError(f"HarborRuntime has no task directory for {task.id!r}")
        env_dir = task_dir / "environment"
        tests_dir = task_dir / "tests"
        if not (env_dir / "Dockerfile").is_file():
            raise FileNotFoundError(f"Harbor task {task.id!r} has no environment/Dockerfile")
        if not (tests_dir / "test.sh").is_file():
            raise FileNotFoundError(f"Harbor task {task.id!r} has no tests/test.sh")

        with tempfile.TemporaryDirectory(prefix=f"hud-harbor-{_slugify(task.id)}-") as tmp:
            tmp_path = Path(tmp)
            workspace = tmp_path / "workspace"
            logs = tmp_path / "logs"
            workspace.mkdir()
            logs.mkdir(parents=True, exist_ok=True)

            image = await self._build_image(env_dir)
            workdir = await _image_workdir(image)
            await _materialize_workspace(image, workspace, workdir)

            compose_file = _compose_file(env_dir)
            if compose_file is not None:
                await _docker("image", "rm", image, check=False)
                acquire = self._compose_container(
                    task, compose_file, workspace, workdir, tests_dir, logs
                )
            else:
                acquire = self._single_container(task, image, workspace, workdir, tests_dir, logs)
            async with acquire as (container, provider):
                env = self._environment_for(task, task_dir, workspace, workdir, logs, container)
                async with _local(env) as runtime:
                    yield Runtime(
                        runtime.url,
                        params={
                            **runtime.params,
                            "provider": provider,
                            "container": container,
                            "ready_timeout": self.ready_timeout,
                        },
                        config=runtime.config,
                    )

    @contextlib.asynccontextmanager
    async def _single_container(
        self,
        task: Task,
        image: str,
        workspace: Path,
        workdir: str,
        tests_dir: Path,
        logs: Path,
    ) -> AsyncIterator[tuple[str, str]]:
        from hud.eval.runtime import _docker

        container_name = f"hud-harbor-{_slugify(task.id)}-{uuid.uuid4().hex[:8]}"
        out, _ = await _docker(
            "run",
            "--detach",
            "--name",
            container_name,
            "--workdir",
            workdir,
            "--entrypoint",
            "sleep",
            "--volume",
            f"{workspace}:{workdir}",
            "--volume",
            f"{tests_dir}:/tests:ro",
            "--volume",
            f"{logs}:/logs",
            image,
            "infinity",
        )
        container = out.strip()
        try:
            yield container, "harbor"
        finally:
            with contextlib.suppress(Exception):
                await _release_mount_permissions(container, workdir)
            await _docker("rm", "--force", "--volumes", container, check=False)
            await _docker("image", "rm", image, check=False)

    @contextlib.asynccontextmanager
    async def _compose_container(
        self,
        task: Task,
        compose_file: Path,
        workspace: Path,
        workdir: str,
        tests_dir: Path,
        logs: Path,
    ) -> AsyncIterator[tuple[str, str]]:
        from hud.eval.runtime import _docker

        project = f"hud-harbor-{_slugify(task.id)}-{uuid.uuid4().hex[:8]}"
        overlay = workspace.parent / "compose.hud.yaml"
        overlay.write_text(
            _compose_overlay(workspace=workspace, workdir=workdir, tests_dir=tests_dir, logs=logs),
            encoding="utf-8",
            newline="\n",
        )
        compose_args = ("compose", "-f", str(compose_file), "-f", str(overlay), "-p", project)
        container = ""
        try:
            await _docker(*compose_args, "up", "--detach", "--build")
            out, _ = await _docker(*compose_args, "ps", "-q", "main")
            container = out.strip()
            if not container:
                raise RuntimeError(
                    f"docker compose project {project} did not create a main service"
                )
            yield container, "harbor-compose"
        finally:
            if container:
                with contextlib.suppress(Exception):
                    await _release_mount_permissions(container, workdir)
            await _docker(
                *compose_args,
                "down",
                "--volumes",
                "--remove-orphans",
                "--rmi",
                "local",
                check=False,
            )

    async def _build_image(self, env_dir: Path) -> str:
        from hud.eval.runtime import _docker

        tag = f"hud-harbor:{_hash_directory(env_dir)}-{uuid.uuid4().hex[:8]}"
        await _docker("build", "--tag", tag, str(env_dir))
        return tag

    def _environment_for(
        self,
        task: Task,
        task_dir: Path,
        workspace: Path,
        workdir: str,
        logs: Path,
        container: str,
    ) -> Environment:
        env = Environment(task.env)
        workspace_daemon = _DockerWorkspace(workspace, container=container, guest_path=workdir)
        verifier_timeout = _verifier_timeout(task_dir)

        @env.initialize
        async def _up() -> None:
            await workspace_daemon.start()
            env.add_capability(workspace_daemon.capability("shell"))

        @env.shutdown
        async def _down() -> None:
            await workspace_daemon.stop()

        @env.template(id=task.id, description=f"Harbor task {task.id}")
        async def _run_harbor_task() -> AsyncGenerator[Any, Any]:
            answer = yield (task_dir / "instruction.md").read_text(encoding="utf-8")
            yield await self._grade(
                container, workdir, logs, answer, verifier_timeout=verifier_timeout
            )

        return env

    async def _grade(
        self, container: str, workdir: str, logs: Path, answer: Any, *, verifier_timeout: float
    ) -> dict[str, Any]:
        answer_file = logs / "agent_answer.txt"
        answer_file.parent.mkdir(parents=True, exist_ok=True)
        answer_file.write_text("" if answer is None else str(answer), encoding="utf-8")
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "--workdir",
            workdir,
            container,
            "bash",
            "/tests/test.sh",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out_bytes, err_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=verifier_timeout
            )
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return {
                "score": 0.0,
                "isError": True,
                "content": f"Harbor verifier timed out after {verifier_timeout:.0f}s",
                "info": {"verifier_timeout_sec": verifier_timeout},
            }
        out = out_bytes.decode("utf-8", "replace")
        err = err_bytes.decode("utf-8", "replace")
        reward, info = _read_harbor_reward(logs / "verifier")
        info.update(
            {
                "stdout": out[-4000:],
                "stderr": err[-4000:],
            }
        )
        if reward is None:
            return {
                "score": 0.0,
                "isError": True,
                "content": "Harbor verifier did not write reward.json or reward.txt",
                "info": info,
            }
        return {"score": reward, "info": info}


class _DockerWorkspace(Workspace):
    """Workspace SFTP over a host bind mount, shell commands via docker exec."""

    def __init__(self, *args: Any, container: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._container = container

    async def _handle_process(self, process: asyncssh.SSHServerProcess[bytes]) -> None:
        command = process.command or "bash -l"
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "-i",
            "--workdir",
            self._guest_path,
            self._container,
            "bash",
            "-lc",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_data, stderr_data = await asyncio.wait_for(proc.communicate(), timeout=3600.0)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            process.stderr.write(b"workspace: command timed out after 3600s\n")
            process.exit(1)
            return
        except asyncio.CancelledError:
            proc.kill()
            await proc.wait()
            raise

        if stdout_data:
            process.stdout.write(stdout_data)
        if stderr_data:
            process.stderr.write(stderr_data)
        process.exit(proc.returncode if proc.returncode is not None else 0)


_DEFAULT_VERIFIER_TIMEOUT = 600.0


def _verifier_timeout(task_dir: Path) -> float:
    """The task's ``[verifier] timeout_sec``, or the Harbor default."""
    try:
        config: dict[str, Any] = tomllib.loads((task_dir / "task.toml").read_text("utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return _DEFAULT_VERIFIER_TIMEOUT
    verifier = config.get("verifier")
    timeout = verifier.get("timeout_sec") if isinstance(verifier, dict) else None
    if isinstance(timeout, int | float) and not isinstance(timeout, bool) and timeout > 0:
        return float(timeout)
    return _DEFAULT_VERIFIER_TIMEOUT


async def _image_workdir(image: str) -> str:
    """The image's configured ``WORKDIR``, or ``/app`` when it declares none."""
    from hud.eval.runtime import _docker

    out, _ = await _docker("image", "inspect", "--format", "{{.Config.WorkingDir}}", image)
    return out.strip() or "/app"


async def _materialize_workspace(image: str, workspace: Path, workdir: str) -> None:
    """Copy the built image's ``workdir`` onto the host workspace, then own it.

    The ``workdir`` bind mount would otherwise shadow everything the Docker build
    generated there (start scripts, installed dependencies, compiled output,
    seeded databases). Copying the image's actual ``workdir`` out first makes the
    mounted workspace a faithful, editable copy of what the image runs. Files
    arrive owned by the container's build user; hand them to the host user so the
    agent can edit them over SFTP and teardown can remove them.
    """
    from hud.eval.runtime import _docker

    out, _ = await _docker("create", image, "true")
    container = out.strip()
    try:
        await _docker("cp", f"{container}:{workdir}/.", str(workspace))
    finally:
        await _docker("rm", "--force", "--volumes", container, check=False)

    if hasattr(os, "getuid"):
        await _docker(
            "run",
            "--rm",
            "--volume",
            f"{workspace}:{workdir}",
            image,
            "chown",
            "-R",
            f"{os.getuid()}:{os.getgid()}",
            workdir,
            check=False,
        )


def _read_harbor_reward(verifier_logs: Path) -> tuple[float | None, dict[str, Any]]:
    reward_json = verifier_logs / "reward.json"
    if reward_json.is_file():
        data = json.loads(reward_json.read_text(encoding="utf-8"))
        if isinstance(data, int | float):
            return float(data), {"reward_file": str(reward_json)}
        if isinstance(data, dict):
            for key in ("reward", "score"):
                value = data.get(key)
                if isinstance(value, int | float):
                    return float(value), {"reward_file": str(reward_json), "reward_json": data}
        return None, {"reward_file": str(reward_json), "reward_parse_error": "no numeric reward"}

    reward_txt = verifier_logs / "reward.txt"
    if reward_txt.is_file():
        text = reward_txt.read_text(encoding="utf-8").strip()
        try:
            return float(text), {"reward_file": str(reward_txt)}
        except ValueError:
            return None, {"reward_file": str(reward_txt), "reward_parse_error": text}

    return None, {}


async def _release_mount_permissions(container: str, workdir: str) -> None:
    """Let the host user delete files that container-root created in mounts."""
    from hud.eval.runtime import _docker

    await _docker(
        "exec",
        container,
        "sh",
        "-lc",
        f"chmod -R a+rwX {workdir} /logs 2>/dev/null || true",
        check=False,
    )


def _compose_file(env_dir: Path) -> Path | None:
    for name in ("docker-compose.yaml", "docker-compose.yml", "compose.yaml", "compose.yml"):
        path = env_dir / name
        if path.is_file():
            return path
    return None


def _compose_overlay(*, workspace: Path, workdir: str, tests_dir: Path, logs: Path) -> str:
    """Compose override that keeps Harbor's main service idle for agent work.

    Only ``main`` is touched: it is parked on ``sleep`` with the materialized
    workspace mounted over its working directory, and the Harbor ``/tests`` and
    ``/logs`` paths bound in. Every other service (databases, caches) is
    inherited from the task's own compose file unchanged.
    """
    return "\n".join(
        [
            "services:",
            "  main:",
            f"    working_dir: {json.dumps(workdir)}",
            '    entrypoint: ["sleep"]',
            '    command: ["infinity"]',
            "    volumes:",
            f"      - {json.dumps(f'{workspace}:{workdir}')}",
            f"      - {json.dumps(f'{tests_dir}:/tests:ro')}",
            f"      - {json.dumps(f'{logs}:/logs')}",
            "",
        ],
    )
