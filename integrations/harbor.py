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
roundtrip. Like every row, the result is runnable
once a placement is supplied (``runtime=Runtime(url)`` against a served substrate
today). Providers receive the row being placed, so a docker provider that
builds and runs each row's ``environment/`` image is the named follow-up —
expressible without engine changes.

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

import hashlib
import json
import logging
import re
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hud.environment import Environment
from hud.environment.server import TaskRunner
from hud.eval import Task, Taskset

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
    root = Path(path)
    if _is_harbor_task(root):
        return True
    if root.is_dir():
        return any(_is_harbor_task(d) for d in root.iterdir() if d.is_dir())
    return False


def load(path: str | Path) -> Taskset:
    """Load a Harbor task dir (or dataset dir) into a :class:`Taskset`.

    One row per task dir (``id`` = the dir name, ``task.toml`` ``[metadata]``
    as columns); rows share one env name per distinct ``environment/`` build
    context (content-hashed), derived from the dataset name.
    """
    root = Path(path).resolve()
    if _is_harbor_task(root):
        task_dirs = [root]
        dataset_name = root.parent.name
    else:
        task_dirs = sorted(d for d in root.iterdir() if d.is_dir() and _is_harbor_task(d))
        dataset_name = root.name
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
        tasks.extend(Task(env=env_name, id=harbor_task.task_id) for harbor_task in group)
    return Taskset(base_name, tasks)


def _slugify(name: str) -> str:
    """A valid env name (lowercase ``[a-z0-9-]``) from a dataset dir name."""
    normalized = re.sub(r"[^a-z0-9-]", "", name.strip().lower().replace(" ", "-").replace("_", "-"))
    return re.sub(r"-+", "-", normalized).strip("-") or "harbor"


def _read_task_config(task_dir: Path) -> dict[str, Any] | None:
    try:
        return tomllib.loads((task_dir / "task.toml").read_text("utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None


def _task_has_instruction(task_dir: Path, config: dict[str, Any]) -> bool:
    """A task carries its instruction at the root (single-step) or under each
    ``steps/<name>/`` directory declared by a ``[[steps]]`` array (multi-step).
    A multi-step task has no root ``instruction.md``, so both forms count."""
    return (task_dir / "instruction.md").is_file() or bool(config.get("steps"))


def _is_harbor_task(path: Path) -> bool:
    if not path.is_dir() or not (path / "task.toml").exists():
        return False
    config = _read_task_config(path)
    if config is None:
        # An unparseable task.toml still identifies a single-step task by its
        # root instruction.md; a multi-step task can only be told apart via the
        # config, so drop it when the config is unreadable.
        return (path / "instruction.md").is_file()
    return _task_has_instruction(path, config)


def _hash_directory(path: Path) -> str:
    """Content-hash a directory for grouping tasks by identical environments."""
    hasher = hashlib.sha256()
    if not path.exists():
        return "empty"
    for file_path in sorted(path.rglob("*")):
        if file_path.is_file():
            hasher.update(str(file_path.relative_to(path)).encode())
            hasher.update(file_path.read_bytes())
    return hasher.hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class _HarborTask:
    """One parsed Harbor task dir."""

    task_id: str
    config: dict[str, Any]
    env_hash: str


def _parse_task(task_dir: Path) -> _HarborTask | None:
    config = _read_task_config(task_dir)
    if config is None:
        # Unparseable config degrades gracefully for a single-step task (kept
        # with an empty config); a multi-step task needs the config to be read.
        if not (task_dir / "instruction.md").is_file():
            LOGGER.warning("failed to parse task.toml in %s", task_dir)
            return None
        LOGGER.warning("failed to parse task.toml in %s; using empty config", task_dir)
        config = {}
    elif not _task_has_instruction(task_dir, config):
        LOGGER.warning("no instruction.md and no [[steps]] in %s", task_dir)
        return None
    env_dir = task_dir / "environment"
    return _HarborTask(
        task_id=task_dir.name,
        config=config,
        env_hash=_hash_directory(env_dir) if env_dir.exists() else "no-env",
    )


# ─── export: HUD tasks -> Harbor task folders ───────────────────────────


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


def _resolve_env(task: Task, authored: dict[str, Environment]) -> Environment:
    """Resolve a task row's env name to a local, authored env defining the task.

    Rows reference envs by name; export materializes prompts in-process, so
    the authored ``Environment`` must be defined in (or next to) the task
    source. A row whose name matches nothing exportable fails loudly.
    """
    env = authored.get(task.env)
    if env is None or task.id not in env.tasks:
        raise TypeError(
            f"harbor export needs a local env defining task {task.id!r} "
            f"(an env.py named {task.env!r} next to the tasks); none was found.",
        )
    return env


# ─── generated files ───────────────────────────────────────────────────

_ENTRYPOINT_SH = """\
#!/bin/sh
# Baked ENTRYPOINT (POSIX sh — slim base images have no bash): serve the HUD
# control channel, run the task setup (parking the paused run), then exec the
# container command. Harbor overrides CMD with `sleep infinity`, so setup must
# run via ENTRYPOINT; `exec "$@"` keeps the channel alive alongside it. The
# agent and the verifier both run in this same container, so the verifier
# reaches the parked run on 127.0.0.1:{port} to grade.
set -u

hud serve env:env --port {port} &

# Wait for the control channel to accept connections (python is always present).
python3 -c 'import socket, sys, time
port = int(sys.argv[1])
for _ in range(120):
    try:
        socket.create_connection(("127.0.0.1", port), 0.5).close()
        break
    except OSError:
        time.sleep(0.5)' {port} || true

# Run the task setup phase and park the run for grading.
hud task start '{task}' --args '{args_json}' --url tcp://127.0.0.1:{port} >/dev/null 2>&1 || true

exec "$@"
"""

_TEST_SH = """\
#!/bin/sh
# Grade the parked HUD run against the agent's work, writing the Harbor reward.
set -u
mkdir -p /logs/verifier

ANSWER_FILE='{answer_file}'
[ -f "$ANSWER_FILE" ] || : > "$ANSWER_FILE"

if hud task grade '{task}' --args '{args_json}' --answer-file "$ANSWER_FILE" \\
    --url tcp://127.0.0.1:{port} > /logs/verifier/reward.txt 2> /logs/verifier/grade.err; then
    :
else
    echo 0 > /logs/verifier/reward.txt
fi
"""

_INSTRUCTION_SUFFIX = """\

---
When you have finished, write your final answer to `{answer_file}`.
"""


def _adapt_env_dockerfile(content: str) -> str:
    """Neutralize the env's own CMD/ENTRYPOINT and bake the boot ENTRYPOINT.

    ENTRYPOINT (not CMD) because Harbor overrides the container command with
    ``sleep infinity``; our entrypoint runs setup then ``exec "$@"`` into it.
    """
    lines: list[str] = []
    for line in content.splitlines():
        stripped = line.strip().upper()
        if stripped.startswith(("CMD ", "CMD[", "ENTRYPOINT ", "ENTRYPOINT[")):
            lines.append(f"# [hud original] {line}")
        else:
            lines.append(line)
    boot_layer = (
        "\n# ─── HUD → Harbor boot entrypoint ───\n"
        "COPY hud_entrypoint.sh /hud_entrypoint.sh\n"
        "RUN chmod +x /hud_entrypoint.sh\n"
        'ENTRYPOINT ["/hud_entrypoint.sh"]\n'
        "# Default command for standalone `docker run`; Harbor injects its own.\n"
        'CMD ["sh", "-c", "sleep infinity"]\n'
    )
    return "\n".join(lines) + "\n" + boot_layer


def _harbor_task_toml(name: str, task: str, args: dict[str, Any], timeout: float) -> str:
    """A Harbor-native ``task.toml`` (``name``/``version`` required by the registry)."""
    return (
        'version = "1.0"\n'
        f'name = "{name}"\n'
        "\n[metadata]\n"
        f'hud_task = "{task}"\n'
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


def _make_ignore(out_root: Path) -> Callable[[str, list[str]], set[str]]:
    """Ignore the standard caches plus the export output dir (which may live under
    the source dir, e.g. ``./harbor_tasks`` next to ``env.py``)."""
    out_resolved = out_root.resolve()

    def _ignore(dirpath: str, names: list[str]) -> set[str]:
        ignored = set(_BUILD_CONTEXT_IGNORE(dirpath, names))
        base = Path(dirpath)
        ignored.update(n for n in names if (base / n).resolve() == out_resolved)
        return ignored

    return _ignore


def _write_environment(
    task_dir: Path,
    source_dir: Path,
    dockerfile: Path,
    task: str,
    args: dict[str, Any],
    out_root: Path,
) -> None:
    """Copy the env build context into ``environment/`` and bake the boot entrypoint."""
    env_out = task_dir / "environment"
    if env_out.exists():
        shutil.rmtree(env_out)
    shutil.copytree(source_dir, env_out, ignore=_make_ignore(out_root))

    # Drop any copied taskset files and the source Dockerfile name we don't use.
    for stale in env_out.glob("*.json*"):
        stale.unlink()
    for name in ("Dockerfile.hud", "dockerfile"):
        leftover = env_out / name
        if leftover.exists() and leftover.name != "Dockerfile":
            leftover.unlink()

    _write_text(env_out / "Dockerfile", _adapt_env_dockerfile(dockerfile.read_text("utf-8")))
    _write_text(
        env_out / "hud_entrypoint.sh",
        _ENTRYPOINT_SH.format(port=CONTROL_PORT, task=task, args_json=json.dumps(args)),
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
    authored = {
        env.name: env
        for module in iter_modules(scan)
        for env in vars(module).values()
        if isinstance(env, Environment)
    }

    dockerfile = _find_dockerfile(source_dir)
    if dockerfile is None:
        raise FileNotFoundError(
            f"no Dockerfile(.hud) next to {source_dir}; harbor export needs the env's "
            "build context to rebuild the image under Harbor.",
        )

    created: list[Path] = []
    for task in tasks:
        env = _resolve_env(task, authored)
        _check_capabilities(env)

        slug = task.slug or task.default_slug()
        task_dir = out / slug
        (task_dir / "tests").mkdir(parents=True, exist_ok=True)

        prompt = await _materialize_prompt(env, task.id, task.args)
        instruction = prompt + _INSTRUCTION_SUFFIX.format(answer_file=answer_file)
        _write_text(task_dir / "instruction.md", instruction)

        _write_text(
            task_dir / "task.toml",
            _harbor_task_toml(slug, task.id, task.args, timeout_sec),
        )

        _write_environment(task_dir, source_dir, dockerfile, task.id, task.args, out)

        _write_text(
            task_dir / "tests" / "test.sh",
            _TEST_SH.format(
                port=CONTROL_PORT,
                task=task.id,
                args_json=json.dumps(task.args),
                answer_file=answer_file,
            ),
        )

        created.append(task_dir)

    return created


__all__ = [
    "ALLOWED_PROTOCOLS",
    "CONTROL_PORT",
    "DEFAULT_ANSWER_FILE",
    "detect",
    "export",
    "load",
]
