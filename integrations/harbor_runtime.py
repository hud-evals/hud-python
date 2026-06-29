"""Local Docker-backed runtime for Harbor task directories."""

from __future__ import annotations

import contextlib
import json
import shlex
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hud.environment import Environment
from hud.environment.workspace import Workspace
from integrations.harbor_common import _hash_directory, _slugify, _task_dirs

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator

    import asyncssh

    from hud.eval import Task
    from hud.eval.runtime import Runtime


class HarborRuntime:
    """Run Harbor task directories through HUD's local rollout engine.

    The provider builds the Harbor task's ``environment/`` Docker context, runs
    a fresh container with a writable host workspace mounted at ``/app``, and
    serves a small HUD control channel from the host process. If the task ships a
    ``docker-compose.yaml``/``.yml``, the provider starts it with an overlay that
    keeps the ``main`` service idle while preserving sidecars such as databases.
    The agent receives normal HUD SSH/SFTP access; shell commands execute inside
    the main container via ``docker exec`` while file transfer edits the mounted
    host workspace. Grading runs the Harbor ``tests/test.sh`` inside the same
    main container and reads ``/logs/verifier/reward.json`` or ``reward.txt``.
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
        self._image_cache: dict[Path, str] = {}

    @contextlib.asynccontextmanager
    async def __call__(self, task: Task) -> AsyncIterator[Runtime]:
        from hud.eval.runtime import Runtime, _local

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
            shutil.copytree(env_dir, workspace)
            _ensure_start_script(workspace)
            _ensure_dockerfile_created_dirs(workspace)
            preserved_paths = _preserved_image_paths(workspace)
            logs.mkdir(parents=True, exist_ok=True)

            compose_file = _compose_file(env_dir)
            if compose_file is not None:
                async with self._compose_container(
                    task,
                    compose_file,
                    workspace,
                    tests_dir,
                    logs,
                    preserved_paths,
                ) as (
                    container,
                    provider,
                ):
                    env = self._environment_for(task, task_dir, workspace, logs, container)
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
            else:
                async with self._single_container(
                    task,
                    task_dir,
                    workspace,
                    tests_dir,
                    logs,
                    preserved_paths,
                ) as (
                    container,
                    provider,
                ):
                    env = self._environment_for(task, task_dir, workspace, logs, container)
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
        task_dir: Path,
        workspace: Path,
        tests_dir: Path,
        logs: Path,
        preserved_paths: list[str],
    ) -> AsyncIterator[tuple[str, str]]:
        from hud.eval.runtime import _docker

        image = await self._image_for(task_dir)
        env_dir = task_dir / "environment"
        await _restore_image_generated_files(image, workspace)
        container_name = f"hud-harbor-{_slugify(task.id)}-{uuid.uuid4().hex[:8]}"
        preserved_volume_args = [arg for path in preserved_paths for arg in ("--volume", path)]
        out, _ = await _docker(
            "run",
            "--detach",
            "--name",
            container_name,
            "--workdir",
            "/app",
            "--entrypoint",
            "sleep",
            "--volume",
            f"{workspace}:/app",
            "--volume",
            f"{tests_dir}:/tests:ro",
            "--volume",
            f"{logs}:/logs",
            *preserved_volume_args,
            image,
            "infinity",
        )
        container = out.strip()
        try:
            yield container, "harbor"
        finally:
            await _release_mount_permissions(container)
            await _docker("rm", "--force", "--volumes", container, check=False)
            await _docker("image", "rm", image, check=False)
            self._image_cache.pop(env_dir, None)

    @contextlib.asynccontextmanager
    async def _compose_container(
        self,
        task: Task,
        compose_file: Path,
        workspace: Path,
        tests_dir: Path,
        logs: Path,
        preserved_paths: list[str],
    ) -> AsyncIterator[tuple[str, str]]:
        from hud.eval.runtime import _docker

        project = f"hud-harbor-{_slugify(task.id)}-{uuid.uuid4().hex[:8]}"
        overlay = workspace.parent / "compose.hud.yaml"
        overlay.write_text(
            _compose_overlay(
                workspace=workspace,
                tests_dir=tests_dir,
                logs=logs,
                preserved_paths=preserved_paths,
            ),
            encoding="utf-8",
            newline="\n",
        )
        compose_args = ("compose", "-f", str(compose_file), "-f", str(overlay), "-p", project)
        await _docker(*compose_args, "up", "--detach", "--build")
        out, _ = await _docker(*compose_args, "ps", "-q", "main")
        container = out.strip()
        if not container:
            raise RuntimeError(f"docker compose project {project} did not create a main service")
        try:
            yield container, "harbor-compose"
        finally:
            await _release_mount_permissions(container)
            await _docker(
                *compose_args,
                "down",
                "--volumes",
                "--remove-orphans",
                "--rmi",
                "local",
                check=False,
            )

    async def _image_for(self, task_dir: Path) -> str:
        from hud.eval.runtime import _docker

        env_dir = task_dir / "environment"
        cached = self._image_cache.get(env_dir)
        if cached is not None:
            return cached
        tag = f"hud-harbor:{_hash_directory(env_dir)}"
        await _docker("build", "--tag", tag, str(env_dir))
        self._image_cache[env_dir] = tag
        return tag

    def _environment_for(
        self,
        task: Task,
        task_dir: Path,
        workspace: Path,
        logs: Path,
        container: str,
    ) -> Environment:
        env = Environment(task.env)
        workspace_daemon = _DockerWorkspace(workspace, container=container, guest_path="/app")

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
            yield await self._grade(container, logs, answer)

        return env

    async def _grade(self, container: str, logs: Path, answer: Any) -> dict[str, Any]:
        from hud.eval.runtime import _docker

        answer_file = logs / "agent_answer.txt"
        answer_file.parent.mkdir(parents=True, exist_ok=True)
        answer_file.write_text("" if answer is None else str(answer), encoding="utf-8")
        out, err = await _docker(
            "exec",
            "--workdir",
            "/app",
            container,
            "bash",
            "/tests/test.sh",
            check=False,
        )
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
        import asyncio

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
            numeric = [float(value) for value in data.values() if isinstance(value, int | float)]
            if numeric:
                return sum(numeric) / len(numeric), {
                    "reward_file": str(reward_json),
                    "reward_json": data,
                }
        return None, {"reward_file": str(reward_json), "reward_parse_error": "no numeric reward"}

    reward_txt = verifier_logs / "reward.txt"
    if reward_txt.is_file():
        text = reward_txt.read_text(encoding="utf-8").strip()
        try:
            return float(text), {"reward_file": str(reward_txt)}
        except ValueError:
            return None, {"reward_file": str(reward_txt), "reward_parse_error": text}

    return None, {}


async def _release_mount_permissions(container: str) -> None:
    """Let the host user delete files that container-root created in mounts."""
    from hud.eval.runtime import _docker

    await _docker(
        "exec",
        container,
        "sh",
        "-lc",
        "chmod -R a+rwX /app /logs 2>/dev/null || true",
        check=False,
    )


def _compose_file(env_dir: Path) -> Path | None:
    for name in ("docker-compose.yaml", "docker-compose.yml", "compose.yaml", "compose.yml"):
        path = env_dir / name
        if path.is_file():
            return path
    return None


def _compose_overlay(
    *,
    workspace: Path,
    tests_dir: Path,
    logs: Path,
    preserved_paths: list[str] | None = None,
) -> str:
    """Compose override that keeps Harbor's main service idle for agent work."""
    preserved_paths = preserved_paths or []
    volume_lines = [
        f"      - {json.dumps(f'{workspace}:/app')}",
        f"      - {json.dumps(f'{tests_dir}:/tests:ro')}",
        f"      - {json.dumps(f'{logs}:/logs')}",
    ]
    volume_lines.extend(f"      - {json.dumps(path)}" for path in preserved_paths)
    return "\n".join(
        [
            "services:",
            "  main:",
            "    build:",
            f"      context: {json.dumps(str(workspace))}",
            "    working_dir: /app",
            '    entrypoint: ["sleep"]',
            '    command: ["infinity"]',
            "    volumes:",
            *volume_lines,
            "",
        ],
    )


def _preserved_image_paths(workspace: Path) -> list[str]:
    """Image-populated subpaths that should survive the editable ``/app`` mount."""
    paths: list[str] = []
    if (workspace / "package.json").is_file():
        paths.append("/app/node_modules")
        if _node_build_output_is_image_populated(workspace, "dist"):
            paths.append("/app/dist")
    if (workspace / "composer.json").is_file():
        paths.append("/app/vendor")
    return paths


def _node_build_output_is_image_populated(workspace: Path, dirname: str) -> bool:
    if (workspace / dirname).exists():
        return False
    dockerfile = workspace / "Dockerfile"
    if not dockerfile.is_file():
        return False
    dockerfile_text = dockerfile.read_text(encoding="utf-8")
    entrypoint = workspace / "docker-entrypoint.sh"
    entrypoint_text = entrypoint.read_text(encoding="utf-8") if entrypoint.is_file() else ""
    return (
        "npm run build" in dockerfile_text
        or f"/app/{dirname}" in dockerfile_text
        or f" {dirname}/" in entrypoint_text
        or f" {dirname}" in entrypoint_text
    )


def _ensure_start_script(workspace: Path) -> None:
    """Preserve build-generated /app/start_app.sh hidden by the workspace mount."""
    start = workspace / "start_app.sh"
    entrypoint = workspace / "docker-entrypoint.sh"
    if not entrypoint.is_file():
        _restore_dockerfile_script(workspace, entrypoint, "/app/docker-entrypoint.sh")
    if entrypoint.is_file():
        entrypoint.chmod(entrypoint.stat().st_mode | 0o111)
    if start.exists():
        start.chmod(start.stat().st_mode | 0o111)
        return
    text = _script_from_dockerfile(workspace, "/app/start_app.sh")
    if text is None and entrypoint.is_file():
        text = "#!/usr/bin/env bash\nset -e\ncd /app\nexec sh /app/docker-entrypoint.sh\n"
    if text is None:
        return
    start.write_text(text, encoding="utf-8", newline="\n")
    start.chmod(0o755)


def _ensure_dockerfile_created_dirs(workspace: Path) -> None:
    """Recreate simple Dockerfile-created ``/app`` dirs hidden by the bind mount."""
    for path in _dockerfile_created_app_dirs(workspace):
        path.mkdir(parents=True, exist_ok=True)


async def _restore_image_generated_files(image: str, workspace: Path) -> None:
    """Copy selected build-generated files from the image into the workspace.

    Some Harbor images initialize file-backed databases during ``docker build``.
    The editable ``/app`` bind mount hides those generated files, so copy them
    out of the built image before starting the task container.
    """
    container_paths = _dockerfile_declared_generated_app_files(workspace)
    if not container_paths:
        return

    from hud.eval.runtime import _docker

    out, _ = await _docker("create", image, "true")
    container = out.strip()
    try:
        for container_path in container_paths:
            host_path = _host_path_for_app_file(workspace, container_path)
            if host_path is None or host_path.exists():
                continue
            host_path.parent.mkdir(parents=True, exist_ok=True)
            await _docker("cp", f"{container}:{container_path}", str(host_path), check=False)
    finally:
        await _docker("rm", "--force", "--volumes", container, check=False)


def _dockerfile_declared_generated_app_files(workspace: Path) -> list[str]:
    """Find Dockerfile-declared file-backed DB paths under ``/app``."""
    dockerfile = workspace / "Dockerfile"
    if not dockerfile.is_file():
        return []

    paths: list[str] = []
    for instruction in _dockerfile_logical_lines(dockerfile.read_text(encoding="utf-8")):
        stripped = instruction.strip()
        if not stripped.startswith("ENV "):
            continue
        for key, value in _env_pairs(stripped.removeprefix("ENV ").strip()):
            if not _is_generated_db_env_key(key):
                continue
            if _is_app_database_path(value):
                paths.append(value)
    return list(dict.fromkeys(paths))


def _env_pairs(body: str) -> list[tuple[str, str]]:
    try:
        tokens = shlex.split(body)
    except ValueError:
        return []
    if not tokens:
        return []

    pairs: list[tuple[str, str]] = []
    if all("=" in token for token in tokens):
        for token in tokens:
            key, value = token.split("=", 1)
            pairs.append((key, value))
        return pairs

    if len(tokens) >= 2:
        pairs.append((tokens[0], tokens[1]))
    return pairs


def _is_generated_db_env_key(key: str) -> bool:
    normalized = key.upper()
    return normalized in {
        "DB_PATH",
        "DATABASE_PATH",
        "SQLITE_PATH",
        "SQLITE_DB_PATH",
        "SQLITE_DATABASE_PATH",
    } or normalized.endswith(("_DB_PATH", "_DATABASE_PATH", "_SQLITE_PATH"))


def _is_app_database_path(path: str) -> bool:
    lowered = path.lower()
    return lowered.startswith("/app/") and lowered.endswith((".db", ".sqlite", ".sqlite3"))


def _host_path_for_app_file(workspace: Path, container_path: str) -> Path | None:
    if not container_path.startswith("/app/"):
        return None
    rel = container_path.removeprefix("/app/")
    if rel.startswith("../") or "/../" in rel or rel == "..":
        return None
    return workspace / rel


def _dockerfile_created_app_dirs(workspace: Path) -> list[Path]:
    dockerfile = workspace / "Dockerfile"
    if not dockerfile.is_file():
        return []
    paths: list[Path] = []
    for instruction in _dockerfile_logical_lines(dockerfile.read_text(encoding="utf-8")):
        stripped = instruction.strip()
        if not stripped.startswith("RUN "):
            continue
        command = stripped.removeprefix("RUN ").strip()
        try:
            tokens = shlex.split(command)
        except ValueError:
            continue
        index = 0
        while index < len(tokens):
            if tokens[index] != "mkdir":
                index += 1
                continue
            index += 1
            while index < len(tokens):
                token = tokens[index]
                if token in {"&&", "||", ";"}:
                    break
                if token.startswith("-"):
                    index += 1
                    continue
                host_path = _app_dir_from_mkdir_token(workspace, token)
                if host_path is not None:
                    paths.append(host_path)
                index += 1
    return paths


def _app_dir_from_mkdir_token(workspace: Path, token: str) -> Path | None:
    if not token or any(char in token for char in "$*?["):
        return None
    raw = token.rstrip("/")
    if raw in {"", "."}:
        return None
    if raw.startswith("/app/"):
        rel = raw.removeprefix("/app/")
    elif raw == "/app":
        return workspace
    elif raw.startswith("/"):
        return None
    else:
        rel = raw
    if rel.startswith("../") or "/../" in rel or rel == "..":
        return None
    return workspace / rel


def _restore_dockerfile_script(workspace: Path, host_path: Path, container_path: str) -> None:
    """Restore a Dockerfile-generated script hidden by a bind mount."""
    text = _script_from_dockerfile(workspace, container_path)
    if text is None:
        return
    host_path.write_text(text, encoding="utf-8", newline="\n")
    host_path.chmod(0o755)


def _script_from_dockerfile(workspace: Path, container_path: str) -> str | None:
    """Extract a Dockerfile-generated script from a simple ``RUN printf`` command."""
    dockerfile = workspace / "Dockerfile"
    if not dockerfile.is_file():
        return None
    for instruction in _dockerfile_logical_lines(dockerfile.read_text(encoding="utf-8")):
        stripped = instruction.strip()
        if not stripped.startswith("RUN ") or container_path not in stripped:
            continue
        command = stripped.removeprefix("RUN ").strip()
        try:
            tokens = shlex.split(command)
        except ValueError:
            continue
        redirect = _redirect_index(tokens, container_path)
        if redirect is None or redirect < 2 or tokens[0] != "printf":
            continue
        text = _script_from_printf_args(tokens[1:redirect])
        if text is not None:
            return text
    return None


def _redirect_index(tokens: list[str], target: str) -> int | None:
    for index, token in enumerate(tokens):
        if token in {">", ">>"} and index + 1 < len(tokens) and tokens[index + 1] == target:
            return index
        if token in {f">{target}", f">>{target}"}:
            return index
    return None


def _script_from_printf_args(args: list[str]) -> str | None:
    if not args:
        return None
    if args[0] in {"%s\\n", "%s\n"}:
        if len(args) < 2:
            return None
        return "\n".join(args[1:]) + "\n"
    if len(args) == 1:
        return args[0].replace("\\r", "\r").replace("\\n", "\n").replace("\\t", "\t")
    return None


def _dockerfile_logical_lines(text: str) -> list[str]:
    """Join backslash-continued Dockerfile lines for simple instruction parsing."""
    lines: list[str] = []
    current = ""
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.endswith("\\"):
            current += line[:-1] + " "
            continue
        lines.append(current + line)
        current = ""
    if current:
        lines.append(current)
    return lines
