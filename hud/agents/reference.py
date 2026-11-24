"""Reference agents that call no model, for checking a task's grading.

``OracleAgent`` runs one of the task's reference solutions, so a sound task grades it as a
pass; ``NopAgent`` does nothing, so a sound task grades it as a fail.

A task's reference solutions are its ``validation`` rows, each a ``Solution``. They travel
with the task, so an environment needs nothing of its own to be checked by the oracle.
"""

from __future__ import annotations

import base64
import binascii
import json
import posixpath
import shlex
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, model_validator

from hud.agents.base import Agent
from hud.agents.cli import WINDOWS_SHELLS
from hud.agents.types import NopConfig, OracleConfig, ToolStep
from hud.types import TextContent, ToolCall, ToolResult
from hud.utils.env_templates import resolve_env_templates
from hud.utils.time import now_iso

if TYPE_CHECKING:
    from hud.capabilities.ssh import SSHClient
    from hud.eval.run import Run
    from hud.eval.task import Task

# A solution's files travel inside the task row, so their total size is capped.
MAX_SOLUTION_BYTES = 1024 * 1024


class SolutionFile(BaseModel):
    """One file ``OracleAgent`` stages under a solution's ``root`` before running its command."""

    model_config = ConfigDict(extra="forbid")

    # Relative to the solution's ``root``.
    path: str = Field(min_length=1)
    content: str
    encoding: Literal["utf-8", "base64"] = "utf-8"
    executable: bool = False

    @model_validator(mode="after")
    def _check(self) -> Self:
        path = PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts or "\\" in self.path:
            raise ValueError(f"solution file path {self.path!r} must be relative to its root")
        if self.encoding == "base64":
            try:
                base64.b64decode(self.content, validate=True)
            except binascii.Error as exc:
                raise ValueError(f"solution file {self.path!r} is not valid base64") from exc
        return self

    @property
    def data(self) -> bytes:
        if self.encoding == "base64":
            return base64.b64decode(self.content)
        return self.content.encode("utf-8")


class Solution(BaseModel):
    """A task's reference solution: one row of its ``validation``.

    ``OracleAgent`` stages ``files`` under ``root``, runs ``command`` in the environment's
    ``ssh`` workspace the way an agent runs commands (as the agent user, from its working
    directory, with ``env``), and submits ``answer``. ``root`` is absolute, as Harbor's
    ``/solution`` is, or relative to the working directory; the agent user must be able
    to create it. ``env`` values may be ``${NAME}`` or ``${NAME:-default}`` templates,
    resolved where the oracle runs.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    files: list[SolutionFile] = Field(default_factory=list[SolutionFile])
    command: str | None = Field(default=None, min_length=1)
    root: str = Field(default="/solution", min_length=1)
    env: dict[str, str] = Field(default_factory=dict[str, str])
    answer: JsonValue = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.command is None and self.answer is None:
            raise ValueError(f"solution {self.name!r} has neither a command nor an answer")
        root = PurePosixPath(self.root)
        if root in (PurePosixPath("/"), PurePosixPath(".")) or ".." in root.parts:
            raise ValueError(
                f"solution {self.name!r} root {self.root!r} must name its own directory"
            )
        if self.command is None and (self.files or self.env):
            raise ValueError(f"solution {self.name!r} has files or env but no command to use them")
        paths = [PurePosixPath(file.path) for file in self.files]
        if len(set(paths)) != len(paths):
            raise ValueError(f"solution {self.name!r} lists a file path twice")
        size = sum(len(file.data) for file in self.files)
        if size > MAX_SOLUTION_BYTES:
            raise ValueError(
                f"solution {self.name!r} files total {size} bytes, over the "
                f"{MAX_SOLUTION_BYTES}-byte limit for a solution carried in its task"
            )
        return self


def task_solution(task: str, validation: list[dict[str, Any]] | None, name: str | None) -> Solution:
    """The solution named ``name`` in a task's ``validation``, or its first when ``name`` is None.

    Raises ``ValueError`` naming ``task`` when it has no solutions, its ``validation``
    rows are not solutions, or none is named ``name``.
    """
    if not validation:
        raise ValueError(
            f"task {task!r} has no reference solution: add one to its validation to run the oracle"
        )
    try:
        solutions = [Solution.model_validate(row) for row in validation]
    except ValidationError as exc:
        raise ValueError(
            f"task {task!r} validation is not a list of reference solutions: {exc}"
        ) from None
    names = [solution.name for solution in solutions]
    if len(set(names)) != len(names):
        raise ValueError(f"task {task!r} has two reference solutions with the same name")
    if name is None:
        return solutions[0]
    for solution in solutions:
        if solution.name == name:
            return solution
    raise ValueError(
        f"task {task!r} has no reference solution named {name!r}; it has {', '.join(names)}"
    )


class OracleAgent(Agent[OracleConfig]):
    """Runs one of the task's reference solutions in an agent's place and submits its answer.

    ``config.solution`` names the solution, else the task's first runs. A task without
    one fails before its environment starts; a command that exits non-zero fails the run.
    """

    config_cls = OracleConfig

    def check_task(self, task: Task) -> None:
        task_solution(task.slug, task.validation, self.config.solution)

    async def __call__(self, run: Run) -> None:
        solution = task_solution(run.task_id, run.validation, self.config.solution)
        if solution.command is not None:
            ssh = cast("SSHClient", await run.client.open("ssh"))
            if ssh.capability.params.get("shell", "bash") in WINDOWS_SHELLS:
                raise RuntimeError("the oracle runs solutions in a POSIX shell, not on Windows")
            await _stage(ssh, solution)
            await _execute(run, ssh, solution, solution.command)
        answer = solution.answer
        run.trace.content = (
            answer if answer is None or isinstance(answer, str) else json.dumps(answer)
        )


async def _stage(ssh: SSHClient, solution: Solution) -> None:
    # A relative root gets a ./ prefix, so no staged path reads as a command option.
    root = PurePosixPath(solution.root)
    root_arg = str(root) if root.is_absolute() else f"./{root}"
    created = await ssh.run(f"mkdir -p {shlex.quote(root_arg)}", check=False, encoding=None)
    if created.returncode != 0:
        raise RuntimeError(
            f"the agent user can't create {solution.root!r} to stage solution "
            f"{solution.name!r}; give the solution a root the agent can write"
            + _detail(created.stderr)
        )
    for file in solution.files:
        path = f"{root_arg.rstrip('/')}/{file.path}"
        target = shlex.quote(path)
        command = f"mkdir -p {shlex.quote(posixpath.dirname(path))} && base64 -d > {target}"
        if file.executable:
            command += f" && chmod +x {target}"
        staged = await ssh.run(
            command, input=base64.b64encode(file.data).decode("ascii"), check=False
        )
        if staged.returncode != 0:
            raise RuntimeError(
                f"staging {file.path!r} of solution {solution.name!r} under {solution.root!r} "
                f"failed" + _detail(staged.stderr)
            )


async def _execute(run: Run, ssh: SSHClient, solution: Solution, command: str) -> None:
    env = {"DEBIAN_FRONTEND": "noninteractive", **resolve_env_templates(solution.env)}
    line = shlex.join(
        ["env", *(f"{key}={value}" for key, value in env.items()), "sh", "-c", command]
    )
    started_at = now_iso()
    result = await ssh.run(line, check=False, encoding=None)
    stdout, stderr = _text(result.stdout), _text(result.stderr)
    failed = result.returncode != 0
    run.record(
        ToolStep(
            call=ToolCall(name="solution", arguments={"name": solution.name, "command": command}),
            result=ToolResult(
                content=[TextContent(type="text", text=stdout + stderr)],
                structuredContent={
                    "exit_code": result.returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                },
                isError=failed,
            ),
            started_at=started_at,
            ended_at=now_iso(),
        )
    )
    if failed:
        raise RuntimeError(
            f"solution {solution.name!r} exited with status {result.returncode}" + _detail(stderr)
        )


def _detail(stderr: bytes | str | None) -> str:
    text = _text(stderr).strip()[-2000:]
    return f": {text}" if text else ""


def _text(value: bytes | str | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value or ""


class NopAgent(Agent[NopConfig]):
    """Submits no answer and leaves the environment untouched."""

    config_cls = NopConfig

    async def __call__(self, run: Run) -> None:
        return
