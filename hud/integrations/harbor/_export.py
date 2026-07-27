"""Export HUD tasks to self-contained Harbor task folders.

The reverse direction: a HUD task source becomes ``task.toml`` +
``instruction.md`` + ``environment/`` + ``tests/test.sh``. Convertible iff
the env's capabilities are ``ssh``/``mcp`` only (Harbor is shell-centric;
``rfb``/``cdp`` don't map). The exported image bakes an ENTRYPOINT that
serves the env control channel and runs the task's setup, parking the run so
``tests/test.sh`` can grade it over the channel — so it depends on that
ENTRYPOINT, not on a Harbor-native verifier.
"""

from __future__ import annotations

import json
import shlex
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hud.environment import Environment
from hud.environment.server import TaskRunner
from hud.eval import Task, Taskset

from ._load import dockerfile_instructions, final_stage, slugify

if TYPE_CHECKING:
    from collections.abc import Callable

#: Capability protocols that map onto Harbor's shell/tool model.
ALLOWED_PROTOCOLS = ("ssh", "mcp")

#: Where the agent writes its final answer (the contract between the
#: instruction and the verifier). Matches the Workspace default guest path.
DEFAULT_ANSWER_FILE = "/workspace/answer.txt"

#: Port the in-container env control channel is served on.
CONTROL_PORT = 8765

#: Build-context entries never copied into the Harbor ``environment/`` dir.
_BUILD_CONTEXT_IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", ".git", ".venv", "venv", "*.egg-info", ".pytest_cache"
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


def _safe_component(slug: str) -> str:
    """A slug reduced to one path component that stays inside the output dir.

    Separators become hyphens so a namespaced id ("suite/fix") stays one
    folder that ``harbor.load()`` can see. A slug with nothing nameable in it
    ("..") is refused rather than silently renamed to a default.
    """
    if not any(character.isalnum() for character in slug):
        raise ValueError(f"task slug {slug!r} does not form a usable directory name")
    return slugify(slug.replace("/", "-").replace("\\", "-"))


async def _materialize_prompt(env: Environment, task: str, args: dict[str, Any]) -> str:
    """Run a task's first yield locally to get its concrete prompt (deterministic).

    The environment is started first: serving runs ``@env.initialize`` before
    any task, so a template reading hook-published state would otherwise see
    an uninitialized environment here and bake a different prompt than the
    exported container produces.
    """
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

hud serve {serve_target} --port {port} &

# Wait for the control channel to accept connections (python is always present).
# A container that never serves, or a task that will not start, is broken
# infrastructure — failing here is honest, where continuing would let the
# verifier score the run 0 as though the agent had simply failed.
python3 -c 'import socket, sys, time
port = int(sys.argv[1])
for _ in range(120):
    try:
        socket.create_connection(("127.0.0.1", port), 0.5).close()
        sys.exit(0)
    except OSError:
        time.sleep(0.5)
sys.exit(1)' {port} || {{
    echo "hud: control channel never came up on port {port}" >&2
    exit 1
}}

# Run the task setup phase and park the run for grading.
hud task start {task} --args {args_json} --url tcp://127.0.0.1:{port} || {{
    echo "hud: task setup failed; refusing to run the agent against an unset task" >&2
    exit 1
}}

exec "$@"
"""

_TEST_SH = """\
#!/bin/sh
# Grade the parked HUD run against the agent's work, writing the Harbor reward.
set -u
mkdir -p /logs/verifier

ANSWER_FILE={answer_file}
[ -f "$ANSWER_FILE" ] || : > "$ANSWER_FILE"

# A grader that cannot run is not a score of 0: exiting nonzero lets Harbor
# record the trial as errored instead of as an agent that failed the task.
if ! hud task grade {task} --args {args_json} --answer-file "$ANSWER_FILE" \\
    --url tcp://127.0.0.1:{port} > /logs/verifier/reward.txt 2> /logs/verifier/grade.err; then
    rm -f /logs/verifier/reward.txt
    echo "hud: grading failed; see /logs/verifier/grade.err" >&2
    exit 1
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
    lines = content.splitlines()
    # A CMD/ENTRYPOINT may span backslash-continued lines; commenting only the
    # first would leave the rest as invalid top-level instructions.
    neutralized = {
        number
        for word, _, numbers in dockerfile_instructions(content)
        if word in ("CMD", "ENTRYPOINT")
        for number in numbers
    }
    lines = [
        f"# [hud original] {line}" if index in neutralized else line
        for index, line in enumerate(lines)
    ]
    # COPY writes a root-owned file, so chmod needs root — and the image's
    # own runtime identity is restored afterwards.
    source_user = final_stage(content).user
    boot_layer = (
        "\n# ─── HUD → Harbor boot entrypoint ───\n"
        "USER root\n"
        "COPY hud_entrypoint.sh /hud_entrypoint.sh\n"
        "RUN chmod +x /hud_entrypoint.sh\n"
        + (f"USER {source_user}\n" if source_user else "")
        + 'ENTRYPOINT ["/hud_entrypoint.sh"]\n'
        "# Default command for standalone `docker run`; Harbor injects its own.\n"
        'CMD ["sh", "-c", "sleep infinity"]\n'
    )
    return "\n".join(lines) + "\n" + boot_layer


def _harbor_task_toml(name: str, task: str, args: dict[str, Any], timeout: float) -> str:
    """A Harbor-native ``task.toml`` (``name``/``version`` required by the registry)."""
    return (
        'version = "1.0"\n'
        f"name = {json.dumps(name)}\n"
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
    taskset_file: Path | None,
    serve_target: str,
    task: str,
    args: dict[str, Any],
    out_root: Path,
) -> None:
    """Copy the env build context into ``environment/`` and bake the boot entrypoint."""
    env_out = task_dir / "environment"
    if env_out.exists():
        shutil.rmtree(env_out)
    shutil.copytree(source_dir, env_out, ignore=_make_ignore(out_root), symlinks=True)

    # Drop the copied taskset file itself — and only that: a build context may
    # legitimately need its own JSON (package.json, tsconfig.json), and a
    # ``.py`` source stays because the exported image serves it.
    if taskset_file is not None:
        copied = env_out / taskset_file.name
        if copied.is_file():
            copied.unlink()
    dockerignore = env_out / ".dockerignore"
    if dockerignore.is_file():
        # The task's ignore rules were written for its own build; they must
        # not exclude the entrypoint this export generates (re-including a
        # path needs its own rule — a directory rule does not cover it).
        dockerignore.write_text(
            dockerignore.read_text("utf-8") + "\n!hud_entrypoint.sh\n",
            encoding="utf-8",
            newline="\n",
        )

    for name in ("Dockerfile.hud", "dockerfile"):
        leftover = env_out / name
        if leftover.exists() and leftover.name != "Dockerfile":
            leftover.unlink()

    _write_text(env_out / "Dockerfile", _adapt_env_dockerfile(dockerfile.read_text("utf-8")))
    _write_text(
        env_out / "hud_entrypoint.sh",
        _ENTRYPOINT_SH.format(
            port=CONTROL_PORT,
            serve_target=shlex.quote(serve_target),
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

    The task's setup runs twice: once here, to capture ``instruction.md``,
    and again inside the exported container at boot, where the run that gets
    graded is parked. A task whose setup is not deterministic for its args
    (randomized challenges, per-run state) will therefore grade a different
    run than the one whose prompt was captured — such tasks need their
    randomness moved into ``args`` before export.

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
    authored: dict[str, Environment] = {}
    # Remember the module attribute each env was found under: the exported
    # container serves *that* target, not a guessed ``env:env``.
    serve_targets: dict[str, str] = {}
    for module in iter_modules(scan):
        module_file = Path(getattr(module, "__file__", "") or "")
        for attribute, value in vars(module).items():
            if isinstance(value, Environment):
                authored[value.name] = value
                serve_targets[value.name] = f"{module_file.stem or 'env'}:{attribute}"

    dockerfile = _find_dockerfile(source_dir)
    if dockerfile is None:
        raise FileNotFoundError(
            f"no Dockerfile(.hud) next to {source_dir}; harbor export needs the env's "
            "build context to rebuild the image under Harbor.",
        )

    created: list[Path] = []
    started: set[str] = set()
    try:
        created = await _export_tasks(
            tasks,
            authored,
            serve_targets,
            started,
            out,
            source_dir,
            src,
            dockerfile,
            answer_file,
            timeout_sec,
        )
    finally:
        for name in started:
            await authored[name].stop()
    return created


async def _export_tasks(
    tasks: list[Task],
    authored: dict[str, Environment],
    serve_targets: dict[str, str],
    started: set[str],
    out: Path,
    source_dir: Path,
    src: Path,
    dockerfile: Path,
    answer_file: str,
    timeout_sec: float,
) -> list[Path]:
    created: list[Path] = []
    claimed: dict[str, str] = {}
    for task in tasks:
        env = _resolve_env(task, authored)
        if env.name not in started:
            # Serving runs @env.initialize before any task; a template reading
            # hook-published state must see the same environment here. Recorded
            # for teardown *before* starting: a hook that raises after an
            # earlier one allocated would otherwise leak it.
            started.add(env.name)
            await env.start()
        _check_capabilities(env)

        # A slug is user data that becomes a directory name: namespaced ids
        # ("suite/fix") would nest out of harbor.load()'s reach and ".." would
        # escape the output directory entirely.
        declared = task.slug or task.default_slug()
        slug = _safe_component(declared)
        if slug in claimed:
            raise ValueError(
                f"task slugs {claimed[slug]!r} and {declared!r} both name the export "
                f"directory {slug!r}; give them distinct slugs"
            )
        claimed[slug] = declared
        task_dir = out / slug
        (task_dir / "tests").mkdir(parents=True, exist_ok=True)

        prompt = await _materialize_prompt(env, task.id, task.args)
        instruction = prompt + _INSTRUCTION_SUFFIX.format(answer_file=answer_file)
        _write_text(task_dir / "instruction.md", instruction)

        _write_text(
            task_dir / "task.toml",
            _harbor_task_toml(slug, task.id, task.args, timeout_sec),
        )

        _write_environment(
            task_dir,
            source_dir,
            dockerfile,
            src if src.is_file() and src.suffix in (".json", ".jsonl") else None,
            serve_targets.get(env.name, "env:env"),
            task.id,
            task.args,
            out,
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
