"""Harbor's lifecycle, run in the controller against its task's boxes."""

from __future__ import annotations

import asyncio
import os
import secrets
import tempfile
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Self
from urllib.parse import urlsplit, urlunsplit

from hud.capabilities.base import Capability
from hud.environment.box import BoxFileTracker, BoxSandbox
from hud.environment.session import session_state_dir
from hud.environment.workspace import Workspace
from hud.runtime.core import RuntimePort, exec_in, resolve_env
from hud.settings import settings
from hud.types import EvaluationResult

from .grading import (
    VERIFIER_LOGS,
    artifact_path,
    env_refs,
    evaluation,
    exclude_artifact_paths,
    verifier_command,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Sequence

    from hud.capabilities.connection import Connection
    from hud.environment.file_tracker import FileTrackingOp
    from hud.runtime.core import Box
    from hud.utils.process import ProcessResult

    from .adapt import MCPServerConfig, TaskConfig
    from .config import Artifact
    from .task import HarborTask, HarborVerifierTask

TESTS = "/tests"
AGENT_ANSWER = "/logs/agent_answer.txt"
# What the agent and the collect hooks leave here, the verifier sees: Harbor's contract.
ARTIFACTS = "/logs/artifacts"
# Where ``ARTIFACTS`` travels in the session state handed to a separate verifier.
LOGS_ARTIFACTS = "logs-artifacts"
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "main"})
# Makes the directory $1 writable by anyone, and each directory above it searchable.
_OPEN_PARENTS = (
    'chmod o+wx "$1" && p="$(dirname "$1")"'
    ' && while [ "$p" != / ]; do chmod o+x "$p" || exit 1; p="$(dirname "$p")"; done'
)


def mcp_port(server: MCPServerConfig, services: Collection[str]) -> RuntimePort | None:
    """The port ``server`` is served from, when one of the task's own ``services`` serves it."""
    if server.url is None or server.transport == "stdio":
        return None
    parsed = urlsplit(server.url)
    host = parsed.hostname
    service = "main" if host in _LOCAL_HOSTS else host if host in services else None
    if service is None:
        return None
    return RuntimePort(
        port=parsed.port or (443 if parsed.scheme == "https" else 80), service=service
    )


def _user(user: str | int | None) -> str | None:
    """A Harbor user spec as the box takes it; ``None`` keeps the image's own user."""
    return None if user is None else str(user)


def _read_text(path: Path) -> str | None:
    return path.read_text("utf-8") if path.is_file() else None


def _frame(result: EvaluationResult) -> dict[str, Any]:
    frame = result.model_dump(mode="json")
    # The core protocol names the reward "score".
    frame["score"] = frame.pop("reward")
    return frame


async def _run(
    box: Box, argv: list[str], *, user: str | None = None, input: bytes = b""
) -> ProcessResult:
    result = await exec_in(box, argv, user=user, input=input)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"command in the Harbor box failed ({result.returncode}): {detail}")
    return result


async def _workdir(box: Box, declared: str | None) -> str:
    """``declared``, or where the box's image starts."""
    return declared or (await _run(box, ["pwd"])).stdout.decode().strip()


async def _verify(
    box: Box,
    task_path: Path,
    config: TaskConfig,
    *,
    cwd: str,
    env: dict[str, str],
    answer: Any,
    owned: Iterable[str] = (),
) -> dict[str, Any]:
    """Copy ``tests/`` into ``box``, run ``test.sh`` as the verifier, and read its reward.

    ``answer`` is at ``AGENT_ANSWER``, and the verifier owns the tests, its logs,
    ``ARTIFACTS`` and ``owned``.
    """
    timeout = config.verifier.timeout_sec or 600.0
    verifier = _user(config.verifier.user)
    identity = await _run(box, ["sh", "-c", 'echo "$(id -u):$(id -g)"'], user=verifier)
    await _run(
        box,
        [
            "sh",
            "-c",
            f"rm -rf {TESTS} {VERIFIER_LOGS} && mkdir -p {VERIFIER_LOGS} && cat > {AGENT_ANSWER}",
        ],
        user="0",
        input=b"" if answer is None else str(answer).encode(),
    )
    tests = task_path / "tests"
    await box.put(tests, TESTS)
    await _run(
        box,
        [
            "sh",
            "-c",
            f'chmod +x {TESTS}/test.sh && owner="$1" && shift && chown -R "$owner" "$@"',
            "sh",
            identity.stdout.decode().strip(),
            TESTS,
            str(VERIFIER_LOGS),
            ARTIFACTS,
            *owned,
        ],
        user="0",
    )
    script = await asyncio.to_thread((tests / "test.sh").read_text, "utf-8")
    execution = await exec_in(
        box,
        verifier_command(script, f"{TESTS}/test.sh"),
        user=verifier,
        cwd=cwd,
        env=env,
        max_wait=timeout,
    )
    with tempfile.TemporaryDirectory(prefix="hud-harbor-") as directory:
        logs = Path(directory) / "verifier"
        await box.get(str(VERIFIER_LOGS), logs)
        reward_json = await asyncio.to_thread(_read_text, logs / "reward.json")
        reward_txt = await asyncio.to_thread(_read_text, logs / "reward.txt")
    return _frame(evaluation(execution, timeout, reward_json, reward_txt))


async def _run_collect_hooks(box: Box, config: TaskConfig) -> None:
    """Run the task's collect hooks, each in the service it names."""
    for hook in config.verifier.collect:
        result = await exec_in(
            box.service(hook.service), ["sh", "-c", hook.command], max_wait=hook.timeout_sec
        )
        if result.timed_out:
            raise TimeoutError(
                f"collect hook on {hook.service!r} timed out after {hook.timeout_sec:g}s"
            )
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", "replace").strip()
            raise RuntimeError(f"collect hook on {hook.service!r} failed: {detail}")


def _stage(target: Path, artifact: Artifact) -> None:
    exclude_artifact_paths(target, artifact.exclude)
    if target.is_symlink() or any(path.is_symlink() for path in target.rglob("*")):
        raise RuntimeError(f"artifact {artifact.source} contains a symbolic link")


class HarborSession:
    """One Harbor task in its box, speaking the core protocol from the controller.

    Starting serves the agent a shell into the box as the task's agent, and the task's MCP
    servers, and returns the instruction. Revoking ends that shell, while what the agent
    left running keeps running, as Harbor's shared verification expects. Grading copies
    ``tests/`` in, runs ``test.sh`` as the verifier, and reads the reward it wrote; with a
    separate verifier, grading runs the collect hooks and copies the artifacts out for it.
    ``${NAME}`` values in the agent's and the verifier's env come from this process's
    environment when the task starts. File tracking follows the agent's workdir as the
    agent sees it, as a served Harbor environment's does, unless that is the filesystem root.
    """

    def __init__(self, task: HarborTask, box: Box) -> None:
        self.task = task
        self.box = box
        self.session_id = "harbor-" + secrets.token_hex(4)
        self._workdir: str | None = None
        self._verifier_env: dict[str, str] = {}
        self._workspace: Workspace[BoxSandbox] | None = None
        self._sandbox: BoxSandbox | None = None
        self._capabilities: list[Capability] = []
        self._files: BoxFileTracker | None = None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.revoke()
        if self._files is not None:
            self._files.close()

    @property
    def capabilities(self) -> list[Capability]:
        return [] if self._workspace is None else self._capabilities

    async def bind(self, connections: Sequence[Connection]) -> None:
        if connections:
            raise ValueError("Harbor tasks take no connections yet")

    async def start_task(self, task_id: str, args: dict[str, Any]) -> dict[str, Any]:
        """Serve the agent its shell and return the instruction; the task is this session's."""
        config = self.task.config
        agent = _user(config.agent.user)
        agent_env = resolve_env(env_refs(config.agent.env), os.environ)
        # Read now, so a missing value fails the task before the agent runs.
        self._verifier_env = resolve_env(env_refs(config.verifier.env), os.environ)
        # Without a declared workdir, the agent starts where the image does.
        workdir = await _workdir(self.box, config.environment.workdir)
        await _run(
            self.box,
            ["sh", "-c", 'rm -rf "$1" && mkdir -p "$1" && chmod 777 "$1"', "sh", ARTIFACTS],
            user="0",
        )
        if settings.file_tracking_enabled and len(PurePosixPath(workdir).parts) > 1:
            self._files = BoxFileTracker(self.box, workdir, user=agent)
        self._sandbox = BoxSandbox(self.box, guest_path=workdir, user=agent, env=agent_env)
        workspace = Workspace(self._sandbox)
        await workspace.start()
        self._workdir, self._workspace = workdir, workspace
        self._capabilities = [workspace.capability(), *self._mcp_capabilities()]
        instruction = await asyncio.to_thread(
            (self.task.path / "instruction.md").read_text, "utf-8"
        )
        return {"prompt": instruction}

    def _mcp_capabilities(self) -> list[Capability]:
        """The task's network MCP servers, at the addresses this process reaches them."""
        capabilities: list[Capability] = []
        for server in self.task.config.environment.mcp_servers:
            if server.url is None or server.transport == "stdio":
                continue
            capability = Capability.mcp(
                name=server.name, url=server.url, transport=server.transport
            )
            if (port := self.task.mcp_ports.get(server.name)) is not None:
                # Reached from here at its published port; from the box, where the task put it.
                parsed = urlsplit(server.url)
                address = self.box.service(port.service).address(port.port)
                url = urlunsplit((parsed.scheme, address, parsed.path, parsed.query, ""))
                capability = replace(
                    Capability.mcp(name=server.name, url=url, transport=server.transport),
                    env_url=capability.url,
                )
            capabilities.append(capability)
        return capabilities

    async def revoke(self) -> None:
        """End the agent's shell; what it started keeps running for grading."""
        workspace, self._workspace = self._workspace, None
        if workspace is not None:
            await workspace.stop()

    async def grade(self, payload: dict[str, Any]) -> dict[str, Any]:
        answer = payload.get("answer")
        if self.task.config.verifier.separate:
            return await self._collect(answer)
        assert self._workdir is not None
        await _run_collect_hooks(self.box, self.task.config)
        return await _verify(
            self.box,
            self.task.path,
            self.task.config,
            cwd=self._workdir,
            env=self._verifier_env,
            answer=answer,
        )

    async def _collect(self, answer: Any) -> dict[str, Any]:
        """Run the collect hooks and gather the artifacts for the separate verifier.

        The declared artifacts and ``ARTIFACTS`` go to this session's state in the box,
        which the runtime hands to the verifier's; the grade carries the answer.
        """
        config = self.task.config
        # What the agent left running is not the verifier's to see.
        assert self._sandbox is not None
        await self._sandbox.terminate_sessions()
        try:
            await _run_collect_hooks(self.box, config)
            with tempfile.TemporaryDirectory(prefix="hud-harbor-artifacts-") as directory:
                staged = Path(directory)
                logs_artifacts = staged / LOGS_ARTIFACTS
                await self.box.get(ARTIFACTS, logs_artifacts)
                if any(path.is_symlink() for path in logs_artifacts.rglob("*")):
                    raise RuntimeError(f"artifact {ARTIFACTS} contains a symbolic link")
                for artifact in config.artifacts:
                    box = self.box.service(artifact.service)
                    source = artifact.source.rstrip("/") or "/"
                    if (await exec_in(box, ["test", "-e", source])).returncode != 0:
                        continue
                    target = artifact_path(artifact, staged / "artifacts")
                    await asyncio.to_thread(target.parent.mkdir, parents=True, exist_ok=True)
                    await box.get(source, target)
                    await asyncio.to_thread(_stage, target, artifact)
                state = str(session_state_dir(self.session_id))
                await _run(
                    self.box, ["sh", "-c", 'rm -rf "$1" && mkdir -p "$1"', "sh", state], user="0"
                )
                await self.box.put(staged, state)
        except Exception as error:
            return _frame(EvaluationResult(isError=True, content=str(error)))
        return _frame(EvaluationResult(info={"answer": "" if answer is None else str(answer)}))

    async def files(self, op: FileTrackingOp) -> dict[str, Any] | None:
        return None if self._files is None else await self._files.call(op)


class HarborVerifierSession:
    """A Harbor task's separate verifier in its own box.

    Grading places the artifacts the agent's box gave up where the task named them, then
    runs ``test.sh`` as the verifier, as a shared verifier does. The verifier's env is the
    verifier environment's and its own, over the image's, over the task environment's.
    """

    def __init__(self, task: HarborVerifierTask, box: Box) -> None:
        self.task = task
        self.box = box
        self.session_id = "harbor-verifier-" + secrets.token_hex(4)
        self.capabilities: list[Capability] = []
        self._environment_env: dict[str, str] = {}
        self._env: dict[str, str] = {}

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def bind(self, connections: Sequence[Connection]) -> None:
        if connections:
            raise ValueError("Harbor tasks take no connections yet")

    async def start_task(self, task_id: str, args: dict[str, Any]) -> dict[str, Any]:
        config = self.task.config
        verifier_environment = config.verifier.environment
        self._environment_env = resolve_env(env_refs(config.environment.env), os.environ)
        self._env = resolve_env(
            env_refs(
                {
                    **(verifier_environment.env if verifier_environment is not None else {}),
                    **config.verifier.env,
                }
            ),
            os.environ,
        )
        return {}

    async def revoke(self) -> None:
        return None

    async def grade(self, payload: dict[str, Any]) -> dict[str, Any]:
        actor = payload["answer"]
        if actor.get("isError"):
            raise RuntimeError(
                actor.get("content") or "the Harbor agent's box gave up no artifacts"
            )
        info = actor["info"]
        # The runtime restored the agent's box's session state as this session's.
        state = session_state_dir(self.session_id)
        artifacts = state / "artifacts"
        if (await exec_in(self.box, ["test", "-d", str(state)])).returncode != 0:
            raise RuntimeError(
                "the Harbor agent's session state did not reach the verifier: its runtime cannot "
                "hand a session to another"
            )
        config = self.task.config
        # ``ARTIFACTS`` as the agent's box left it, where the verifier reads it.
        await _run(
            self.box,
            [
                "sh",
                "-c",
                'rm -rf "$2" && mkdir -p "$2" && cp -a "$1/." "$2/"',
                "sh",
                str(state / LOGS_ARTIFACTS),
                ARTIFACTS,
            ],
            user="0",
        )
        placed: list[str] = []
        for artifact in config.artifacts:
            staged = str(artifact_path(artifact, artifacts))
            if (await exec_in(self.box, ["test", "-e", staged])).returncode != 0:
                continue
            destination = artifact.source.rstrip("/") or "/"
            if destination == "/":
                raise ValueError("the verifier root cannot be replaced by an artifact")
            parent = str(PurePosixPath(destination).parent)
            await _run(
                self.box,
                [
                    "sh",
                    "-c",
                    'rm -rf "$2" && mkdir -p "$3" && cp -a "$1" "$2"',
                    "sh",
                    staged,
                    destination,
                    parent,
                ],
                user="0",
            )
            # The verifier may replace what it was given, so its directory is writable and the
            # way to it searchable.
            await _run(self.box, ["sh", "-c", _OPEN_PARENTS, "sh", parent], user="0")
            placed.append(destination)
        verifier_environment = config.verifier.environment
        # The task environment's env reaches the verifier where its image sets none.
        listed = await _run(self.box, ["env"], user=_user(config.verifier.user))
        image_env = {
            line.partition("=")[0] for line in listed.stdout.decode().splitlines() if "=" in line
        }
        environment_env = {
            key: value for key, value in self._environment_env.items() if key not in image_env
        }
        return await _verify(
            self.box,
            self.task.path,
            config,
            cwd=await _workdir(
                self.box,
                verifier_environment.workdir if verifier_environment is not None else None,
            ),
            env={**environment_env, **self._env},
            answer=info["answer"],
            owned=placed,
        )

    async def files(self, op: FileTrackingOp) -> dict[str, Any] | None:
        return None
