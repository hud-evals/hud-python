"""Export HUD tasks to Harbor task folders.

:func:`export` turns a task source (JSON/JSONL or ``.py``, like ``hud eval``) into
Harbor task folders (``task.toml`` + ``instruction.md`` + ``environment/`` +
``tests/test.sh``). Convertible iff the env's capabilities are ``ssh``/``mcp`` only
(Harbor is shell-centric; ``rfb``/``cdp`` don't map).

Lifecycle mapping (HUD setup/evaluate → Harbor image/verifier):

* The env's build context is copied into ``environment/`` and a ``hud_entrypoint.sh``
  is baked in as the image ENTRYPOINT (Harbor overrides CMD with ``sleep infinity``).
  At container start it serves the env control channel (``hud dev``) and runs the
  task's **setup** (``hud task start``), which parks the paused run on the env so a
  later connection can grade it, then ``exec "$@"`` into the container command.
* The agent then works in the container and writes its answer to ``answer_file``.
* ``tests/test.sh`` runs the task's **evaluate** (``hud task grade``) against the
  parked run and writes the reward to ``/logs/verifier/reward.txt``.

Round-trip note: the exported task grades over the HUD control channel, so it is
*not* a harness-agnostic Harbor task — it depends on the baked ENTRYPOINT serving
that channel. Re-importing it via ``hud convert --from harbor`` does **not**
round-trip the grading: the generated HUD env serves its own ``run-task`` channel
on the same port, and its scenario runs this ``test.sh`` mid-evaluate, so the inner
``hud task grade --url`` collides with the outer channel. The two converters adapt
to different harnesses; they are not inverses.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from hud.environment import Environment

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


def _write_text(path: Path, text: str) -> None:
    """Write a generated file with LF endings (these run in Linux containers;
    the default Windows ``\\r\\n`` translation breaks shebangs and shell scripts)."""
    path.write_text(text, encoding="utf-8", newline="\n")


def _check_capabilities(env: Environment) -> None:
    bad = [
        c.protocol
        for c in env.capabilities
        if c.protocol.split("/", 1)[0] not in ALLOWED_PROTOCOLS
    ]
    if bad:
        raise ValueError(
            f"env {env.name!r} declares non-Harbor capabilities {bad}; "
            f"only {'/'.join(ALLOWED_PROTOCOLS)} are convertible.",
        )


async def _materialize_prompt(env: Environment, task: str, args: dict[str, Any]) -> str:
    """Run a task's first yield locally to get its concrete prompt (deterministic)."""
    from hud.environment.task import TaskRunner

    runner = TaskRunner(env._tasks[task], args)
    try:
        payload = await runner.start()
    finally:
        await runner.cancel()
    prompt = payload.get("prompt")
    return prompt if isinstance(prompt, str) else json.dumps(prompt, indent=2, default=str)


def _resolve_env(variant: Any) -> Environment:
    """Resolve a variant's env-ref to a local :class:`Environment` for materialization.

    A ``Variant`` from a Python source carries the ``Environment`` directly; one
    loaded from a tasks file carries a ``LocalSandbox`` over it (module env-ref).
    Remote / HUD-hosted env-refs can't be materialized locally.
    """
    from hud.environment import Environment
    from hud.eval.sandbox import LocalSandbox

    env = variant.env
    if isinstance(env, LocalSandbox):
        env = env._env
    if not isinstance(env, Environment):
        raise TypeError(
            "harbor export needs a local Environment (a module env-ref or env.py); "
            f"got {type(variant.env).__name__}. Remote/HUD env-refs aren't supported.",
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

hud dev env:env --port {port} &

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
        '# Default command for standalone `docker run`; Harbor injects its own.\n'
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
    for stale in env_out.glob("*.json"):
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
    args}`` entries) or a ``.py`` file/dir exposing ``Variant``s. One folder is
    written per task (task + args), each a self-contained Harbor task. Requires the
    env's build context (a ``Dockerfile.hud``/``Dockerfile`` next to the source).
    Returns the created task directories.
    """
    from hud.cli.utils.collect import collect_variants, load_variants_json

    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    src = Path(source).resolve()
    source_dir = src.parent if src.is_file() else src

    if src.suffix in (".json", ".jsonl"):
        variants = load_variants_json(src)
    else:
        variants = collect_variants(source)

    dockerfile = _find_dockerfile(source_dir)
    if dockerfile is None:
        raise FileNotFoundError(
            f"no Dockerfile(.hud) next to {source_dir}; harbor export needs the env's "
            "build context to rebuild the image under Harbor.",
        )

    created: list[Path] = []
    for variant in variants:
        env = _resolve_env(variant)
        _check_capabilities(env)

        slug = variant.slug or variant.default_slug()
        task_dir = out / slug
        (task_dir / "tests").mkdir(parents=True, exist_ok=True)

        prompt = await _materialize_prompt(env, variant.task, variant.args)
        instruction = prompt + _INSTRUCTION_SUFFIX.format(answer_file=answer_file)
        _write_text(task_dir / "instruction.md", instruction)

        _write_text(
            task_dir / "task.toml",
            _harbor_task_toml(slug, variant.task, variant.args, timeout_sec),
        )

        _write_environment(task_dir, source_dir, dockerfile, variant.task, variant.args, out)

        _write_text(
            task_dir / "tests" / "test.sh",
            _TEST_SH.format(
                port=CONTROL_PORT,
                task=variant.task,
                args_json=json.dumps(variant.args),
                answer_file=answer_file,
            ),
        )

        created.append(task_dir)

    return created


__all__ = ["ALLOWED_PROTOCOLS", "CONTROL_PORT", "DEFAULT_ANSWER_FILE", "export"]
