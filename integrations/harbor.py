"""Harbor integration: load Harbor task dirs as a Taskset; export HUD tasks to Harbor.

Harbor task structure (terminal-bench layout)::

    task_name/
    ├── instruction.md          # agent prompt
    ├── task.toml               # config: timeouts, metadata, resources
    ├── environment/Dockerfile  # container the agent runs in
    ├── tests/test.sh           # verification -> writes reward.txt
    └── solution/               # optional (ignored)

:func:`load` parses a task dir (or a dataset of them) into rows sharing one
env name per distinct ``environment/`` build context — no codegen, no
roundtrip. Like every row, the result is runnable once a placement is supplied.
Use :class:`HarborRuntime` for local Docker-backed execution of Harbor tasks, or
``runtime=Runtime(url)`` to attach to a substrate served elsewhere.

:func:`export` is the reverse direction: turn a HUD task source into
self-contained Harbor task folders (``task.toml`` + ``instruction.md`` +
``environment/`` + ``tests/test.sh``). Convertible iff the env's capabilities
are ``ssh``/``mcp`` only (Harbor is shell-centric; ``rfb``/``cdp`` don't map).

Export lifecycle mapping (HUD setup/evaluate → Harbor image/verifier):

* The env's build context is copied into ``environment/`` and a ``hud_entrypoint.sh``
  is baked in as the image ENTRYPOINT (Harbor overrides CMD with ``sleep infinity``).
  At container start it serves the env control channel (``hud serve``) and runs the
  task's **setup** (``hud task start``), which parks the paused run on the env so a
  later connection can grade it, then ``exec "$@"`` into the container command.
* The agent then works in the container and writes its answer to ``answer_file``.
* ``tests/test.sh`` runs the task's **evaluate** (``hud task grade``) against the
  parked run and writes the reward to ``/logs/verifier/reward.txt``.

The exported task grades over the HUD control channel, so it is *not* a
harness-agnostic Harbor task — it depends on the baked ENTRYPOINT serving that
channel.
"""

from __future__ import annotations

import json
import logging
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hud.environment import Environment
from hud.environment.server import TaskRunner
from hud.eval import Task, Taskset
from integrations.harbor_common import (
    _HarborTask,
    _is_harbor_task,
    _parse_task,
    _slugify,
    _task_dirs,
)
from integrations.harbor_runtime import HarborRuntime

if TYPE_CHECKING:
    from collections.abc import Callable

LOGGER = logging.getLogger(__name__)

#: Capability protocols that map onto Harbor's shell/tool model.
ALLOWED_PROTOCOLS = ("ssh", "mcp")

#: Where the agent writes its final answer (the contract between the instruction
#: and the verifier). Matches the Workspace default guest path.
DEFAULT_ANSWER_FILE = "/workspace/answer.txt"

#: Port the in-container env control channel is served on.
CONTROL_PORT = 8765

#: Build-context entries never copied into the Harbor ``environment/`` dir.
_BUILD_CONTEXT_IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", ".git", ".venv", "venv", "*.egg-info", ".pytest_cache"
)

# ─── load: Harbor dirs -> Taskset ──────────────────────────────────────


def detect(path: str | Path) -> bool:
    """True when *path* is a Harbor task dir or a dataset of them."""
    return bool(_task_dirs(path))


def load(path: str | Path) -> Taskset:
    """Load a Harbor task dir (or dataset dir) into a :class:`Taskset`.

    One row per task dir (``id`` = the dir name, ``task.toml`` ``[metadata]``
    as columns); rows share one env name per distinct ``environment/`` build
    context (content-hashed), derived from the dataset name.
    """
    root = Path(path).resolve()
    task_dirs = _task_dirs(root)
    dataset_name = root.parent.name if _is_harbor_task(root) else root.name
    if not task_dirs:
        raise ValueError(f"no Harbor tasks found in {path}")

    parsed = [task for task_dir in task_dirs if (task := _parse_task(task_dir)) is not None]
    if not parsed:
        raise ValueError(f"all Harbor tasks under {path} failed to parse")
    if len(parsed) < len(task_dirs):
        LOGGER.warning(
            "skipped %d Harbor task(s) that failed to parse", len(task_dirs) - len(parsed)
        )

    groups: dict[str, list[_HarborTask]] = {}
    for harbor_task in parsed:
        groups.setdefault(harbor_task.env_hash, []).append(harbor_task)
    sorted_groups = sorted(groups.values(), key=lambda group: -len(group))

    base_name = _slugify(dataset_name)
    tasks: list[Task] = []
    for idx, group in enumerate(sorted_groups, start=1):
        env_name = base_name if len(sorted_groups) == 1 else f"{base_name}-g{idx}"
        tasks.extend(
            Task(
                env=env_name,
                id=harbor_task.task_id,
                columns=(
                    metadata
                    if isinstance((metadata := harbor_task.config.get("metadata")), dict)
                    else None
                ),
            )
            for harbor_task in group
        )
    return Taskset(base_name, tasks)


# ─── export: HUD tasks -> Harbor task folders ───────────────────────────


@dataclass(frozen=True)
class _AuthoredEnvironment:
    """An authored env together with the source pointer Harbor must serve."""

    env: Environment
    module_path: Path
    symbol: str


def _write_text(path: Path, text: str) -> None:
    """Write a generated file with LF endings (these run in Linux containers;
    the default Windows ``\\r\\n`` translation breaks shebangs and shell scripts)."""
    path.write_text(text, encoding="utf-8", newline="\n")


def _check_capabilities(env: Environment) -> None:
    bad = [
        c.protocol for c in env.capabilities if c.protocol.split("/", 1)[0] not in ALLOWED_PROTOCOLS
    ]
    if bad:
        raise ValueError(
            f"env {env.name!r} declares non-Harbor capabilities {bad}; "
            f"only {'/'.join(ALLOWED_PROTOCOLS)} are convertible.",
        )


async def _materialize_prompt(env: Environment, task: str, args: dict[str, Any]) -> str:
    """Run a task's first yield locally to get its concrete prompt (deterministic)."""
    runner = TaskRunner(env.tasks[task], args)
    try:
        payload = await runner.start()
    finally:
        await runner.cancel()
    prompt = payload.get("prompt")
    return prompt if isinstance(prompt, str) else json.dumps(prompt, indent=2, default=str)


def _resolve_env(
    task: Task,
    authored: dict[str, list[_AuthoredEnvironment]],
) -> _AuthoredEnvironment:
    """Resolve a task row's env name to a local, authored env defining the task.

    Rows reference envs by name; export materializes prompts in-process, so
    the authored ``Environment`` must be defined in (or next to) the task
    source. A row whose name matches nothing exportable fails loudly.
    """
    matches = [ref for ref in authored.get(task.env, ()) if task.id in ref.env.tasks]
    # One Environment may be re-exported under several symbols. Preserve the
    # first concrete pointer, but reject genuinely distinct same-named envs.
    unique = {id(ref.env): ref for ref in reversed(matches)}
    if not unique:
        raise TypeError(
            f"harbor export needs a local env defining task {task.id!r} "
            f"(an Environment named {task.env!r} in the source or an adjacent "
            "Python module); none was found.",
        )
    if len(unique) > 1:
        raise TypeError(
            f"harbor export found multiple local envs named {task.env!r} "
            f"defining task {task.id!r}; make the env name unique.",
        )
    return next(iter(unique.values()))


# ─── generated files ───────────────────────────────────────────────────

_ENTRYPOINT_SH = """\
#!/bin/sh
# Baked ENTRYPOINT (POSIX sh — slim base images have no bash): serve the HUD
# control channel, run the task setup (parking the paused run), then exec the
# container command. Harbor overrides CMD with `sleep infinity`, so setup must
# run via ENTRYPOINT; `exec "$@"` keeps the channel alive alongside it. The
# agent and the verifier both run in this same container, so the verifier
# reaches the parked run on 127.0.0.1:{port} to grade.
set -eu

HUD_ENV_TARGET={env_target}
HUD_TASK={task}
HUD_ARGS={args_json}
HUD_READY_FILE=/tmp/.hud-harbor-ready
server_pid=

cleanup() {{
    status=$?
    trap - EXIT HUP INT TERM
    if [ -n "$server_pid" ]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    exit "$status"
}}

trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

rm -f "$HUD_READY_FILE"

hud serve "$HUD_ENV_TARGET" --port {port} &
server_pid=$!

# Wait for the control channel to accept connections (python is always present).
python3 -c 'import os, socket, sys, time
port = int(sys.argv[1])
pid = int(sys.argv[2])
for _ in range(120):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        raise SystemExit("hud serve exited before becoming ready") from None
    try:
        socket.create_connection(("127.0.0.1", port), 0.5).close()
        break
    except OSError:
        time.sleep(0.5)
else:
    raise SystemExit("hud serve did not become ready in 60 seconds")' {port} "$server_pid"

# Run the task setup phase and park the run for grading.
hud task start --args "$HUD_ARGS" --url tcp://127.0.0.1:{port} -- "$HUD_TASK" >/dev/null
: > "$HUD_READY_FILE"

trap - EXIT HUP INT TERM
exec "$@"
"""

_TEST_SH = """\
#!/bin/sh
# Grade the parked HUD run against the agent's work, writing the Harbor reward.
set -eu
mkdir -p /logs/verifier

HUD_TASK={task}
HUD_ARGS={args_json}
ANSWER_FILE={answer_file}
[ -f "$ANSWER_FILE" ] || : > "$ANSWER_FILE"
REWARD_FILE=/logs/verifier/reward.txt
REWARD_TMP=/logs/verifier/reward.txt.tmp
GRADE_ERR=/logs/verifier/grade.err
rm -f "$REWARD_FILE" "$REWARD_TMP"

if hud task grade --args "$HUD_ARGS" --answer-file "$ANSWER_FILE" \\
    --url tcp://127.0.0.1:{port} -- "$HUD_TASK" \
    > "$REWARD_TMP" 2> "$GRADE_ERR"; then
    mv "$REWARD_TMP" "$REWARD_FILE"
else
    status=$?
    [ ! -s "$GRADE_ERR" ] || cat "$GRADE_ERR" >&2
    rm -f "$REWARD_TMP" "$REWARD_FILE"
    exit "$status"
fi
"""

_INSTRUCTION_SUFFIX = """\

---
When you have finished, write your final answer to `{answer_file}`.
"""


def _adapt_env_dockerfile(content: str) -> str:
    """Append the HUD boot process as the image's final startup configuration.

    Docker uses the final CMD/ENTRYPOINT, so preserving earlier instructions
    verbatim also preserves valid multiline JSON-form instructions. ENTRYPOINT
    runs setup before handing off to Harbor's command; the healthcheck prevents
    Compose ``--wait`` from racing ahead of that setup.
    """
    boot_layer = (
        "\n# ─── HUD → Harbor boot entrypoint ───\n"
        "COPY hud_entrypoint.sh /hud_entrypoint.sh\n"
        "RUN chmod +x /hud_entrypoint.sh\n"
        'ENTRYPOINT ["/hud_entrypoint.sh"]\n'
        "# Default command for standalone `docker run`; Harbor injects its own.\n"
        'CMD ["sh", "-c", "sleep infinity"]\n'
        "HEALTHCHECK --interval=1s --timeout=1s --start-period=1s --retries=120 "
        'CMD ["test", "-f", "/tmp/.hud-harbor-ready"]\n'
    )
    return content.rstrip() + "\n" + boot_layer


def _harbor_task_toml(name: str, task: str, args: dict[str, Any], timeout: float) -> str:
    """Return a current, registry-publishable Harbor ``task.toml``.

    Harbor's package identity lives under ``[task]`` and must use ``org/name``
    form.  A root-level ``name`` is merely ignored by Harbor's ``TaskConfig``;
    the publisher then rejects the task because no package was declared.
    """
    package_name = f"hud/{_slugify(name)}"
    return (
        'schema_version = "1.4"\n'
        "\n[task]\n"
        f"name = {json.dumps(package_name)}\n"
        "\n[metadata]\n"
        f"hud_task = {json.dumps(task)}\n"
        f"hud_args = {json.dumps(json.dumps(args))}\n"
        "\n[agent]\n"
        f"timeout_sec = {timeout}\n"
        "\n[verifier]\n"
        f"timeout_sec = {timeout}\n"
    )


def _find_dockerfile(source_dir: Path) -> Path | None:
    return next(
        (source_dir / n for n in ("Dockerfile.hud", "Dockerfile") if (source_dir / n).exists()),
        None,
    )


def _make_ignore(
    out_root: Path,
    excluded_files: tuple[Path, ...] = (),
) -> Callable[[str, list[str]], set[str]]:
    """Ignore the standard caches plus the export output dir (which may live under
    the source dir, e.g. ``./harbor_tasks`` next to ``env.py``)."""
    out_resolved = out_root.resolve()
    excluded_resolved = {path.resolve() for path in excluded_files}

    def _ignore(dirpath: str, names: list[str]) -> set[str]:
        ignored = set(_BUILD_CONTEXT_IGNORE(dirpath, names))
        base = Path(dirpath)
        ignored.update(
            n
            for n in names
            if (candidate := (base / n).resolve()) == out_resolved or candidate in excluded_resolved
        )
        return ignored

    return _ignore


def _write_environment(
    task_dir: Path,
    source_dir: Path,
    dockerfile: Path,
    task: str,
    args: dict[str, Any],
    env_target: str,
    out_root: Path,
    taskset_source: Path | None,
) -> None:
    """Copy the env build context into ``environment/`` and bake the boot entrypoint."""
    env_out = task_dir / "environment"
    if env_out.exists():
        shutil.rmtree(env_out)
    excluded = (taskset_source,) if taskset_source is not None else ()
    shutil.copytree(source_dir, env_out, ignore=_make_ignore(out_root, excluded))

    # The exact JSON/JSONL taskset source was excluded during copying; unrelated
    # JSON build inputs remain part of the environment context.
    for name in ("Dockerfile.hud", "dockerfile"):
        leftover = env_out / name
        if leftover.exists() and leftover.name != "Dockerfile":
            leftover.unlink()

    _write_text(env_out / "Dockerfile", _adapt_env_dockerfile(dockerfile.read_text("utf-8")))
    _write_text(
        env_out / "hud_entrypoint.sh",
        _ENTRYPOINT_SH.format(
            port=CONTROL_PORT,
            env_target=shlex.quote(env_target),
            task=shlex.quote(task),
            args_json=shlex.quote(json.dumps(args)),
        ),
    )


async def export(
    source: str,
    out_dir: str | Path,
    *,
    answer_file: str = DEFAULT_ANSWER_FILE,
    timeout_sec: float = 600.0,
) -> list[Path]:
    """Export HUD tasks from *source* into Harbor task folders under *out_dir*.

    *source* is either a **tasks file** (``.json`` / ``.jsonl`` of ``{env, task,
    args}`` entries) or a ``.py`` file/dir exposing ``Task``s. One folder is
    written per task (task + args), each a self-contained Harbor task. Requires the
    env's build context (a ``Dockerfile.hud``/``Dockerfile`` next to the source).
    Returns the created task directories.
    """
    from hud.utils.modules import iter_modules

    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    src = Path(source).resolve()
    source_dir = src.parent if src.is_file() else src

    tasks = list(Taskset.from_file(src))
    # Rows reference envs by name; collect the authored envs (defined in the
    # source, or next to a tasks file) to materialize prompts in-process.
    scan = source_dir if src.suffix in (".json", ".jsonl") else src
    authored: dict[str, list[_AuthoredEnvironment]] = {}
    for module in iter_modules(scan):
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        for symbol, env in vars(module).items():
            if isinstance(env, Environment):
                authored.setdefault(env.name, []).append(
                    _AuthoredEnvironment(env, Path(module_file).resolve(), symbol),
                )

    dockerfile = _find_dockerfile(source_dir)
    if dockerfile is None:
        raise FileNotFoundError(
            f"no Dockerfile(.hud) next to {source_dir}; harbor export needs the env's "
            "build context to rebuild the image under Harbor.",
        )

    normalized_tasks = [(_slugify(task.slug or task.default_slug()), task) for task in tasks]
    seen_slugs: set[str] = set()
    for slug, _task in normalized_tasks:
        if slug in seen_slugs:
            raise ValueError(
                f"multiple HUD task slugs normalize to Harbor folder {slug!r}; "
                "give each task a distinct slug.",
            )
        seen_slugs.add(slug)

    created: list[Path] = []
    for slug, task in normalized_tasks:
        authored_env = _resolve_env(task, authored)
        env = authored_env.env
        _check_capabilities(env)

        task_dir = out / slug
        (task_dir / "tests").mkdir(parents=True, exist_ok=True)

        prompt = await _materialize_prompt(env, task.id, task.args)
        instruction = prompt + _INSTRUCTION_SUFFIX.format(answer_file=answer_file)
        _write_text(task_dir / "instruction.md", instruction)

        _write_text(
            task_dir / "task.toml",
            _harbor_task_toml(slug, task.id, task.args, timeout_sec),
        )

        env_module = authored_env.module_path.relative_to(source_dir).as_posix()
        env_target = f"{env_module}:{authored_env.symbol}"
        taskset_source = src if src.suffix in (".json", ".jsonl") else None
        _write_environment(
            task_dir,
            source_dir,
            dockerfile,
            task.id,
            task.args,
            env_target,
            out,
            taskset_source,
        )

        _write_text(
            task_dir / "tests" / "test.sh",
            _TEST_SH.format(
                port=CONTROL_PORT,
                task=shlex.quote(task.id),
                args_json=shlex.quote(json.dumps(task.args)),
                answer_file=shlex.quote(answer_file),
            ),
        )

        created.append(task_dir)

    return created


__all__ = [
    "ALLOWED_PROTOCOLS",
    "CONTROL_PORT",
    "DEFAULT_ANSWER_FILE",
    "HarborRuntime",
    "detect",
    "export",
    "load",
]
