"""Export HUD tasks to Harbor task folders (deterministic, build-time).

:func:`export` turns a HUD task source (a ``Variant`` source, same as ``hud eval``)
into Harbor task folders — ``task.toml`` + ``instruction.md`` + ``environment/`` +
``tests/test.sh``. Driven by the ``hud harbor`` CLI command.

Grading happens at run-time via ``hud client run`` (a CLI over the env control
channel): the generated ``tests/test.sh`` connects to the env served in the
container and submits the agent's answer. Because grading runs in the env that
shares the agent's ``ssh`` workspace, state-based checks see the agent's changes.

A HUD env is Harbor-convertible iff all its capabilities are ``ssh`` and/or ``mcp``
(Harbor is shell/script-centric; ``rfb``/``cdp`` have no Harbor analogue).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hud.environment import Environment

#: Capability protocols that map onto Harbor's shell/tool model.
ALLOWED_PROTOCOLS = ("ssh", "mcp")


def _variant_slug(task: str, args: dict[str, Any]) -> str:
    """Stable per-task folder name: task id, disambiguated by args when present."""
    if not args:
        return task
    digest = hashlib.sha1(  # noqa: S324 - non-crypto, stable disambiguator
        json.dumps(args, sort_keys=True, default=str).encode("utf-8"),
    ).hexdigest()[:8]
    return f"{task}-{digest}"


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


_TEST_SH = """\
#!/usr/bin/env bash
# Grade by driving the env control channel via `hud client run`.
set -euo pipefail
mkdir -p /logs/verifier
hud client run '{task}' \\
    --args '{args_json}' \\
    --answer "$(cat /workspace/answer.txt 2>/dev/null || true)" \\
    > /logs/verifier/reward.txt
"""


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


async def export(source: str, out_dir: str | Path) -> list[Path]:
    """Export HUD tasks from *source* into Harbor task folders under *out_dir*.

    *source* is either a **tasks file** (``.json`` / ``.jsonl`` of ``{env, task,
    args}`` entries — same as ``hud eval``) or a ``.py`` file/dir exposing
    ``Variant``s. One folder is written per task (task + args), each with
    ``task.toml`` / ``instruction.md`` / ``environment/Dockerfile`` / ``tests/test.sh``.
    Returns the created task directories. Deterministic: same env + args ⇒ same output.
    """
    from hud.cli.utils.collect import collect_variants, load_variants_json

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(source).resolve()
    source_dir = src.parent if src.is_file() else src
    if src.suffix in (".json", ".jsonl"):
        variants = load_variants_json(src)
    else:
        variants = collect_variants(source)
    dockerfile = next(
        (source_dir / n for n in ("Dockerfile.hud", "Dockerfile") if (source_dir / n).exists()),
        None,
    )

    created: list[Path] = []
    for variant in variants:
        env = _resolve_env(variant)
        _check_capabilities(env)

        slug = variant.slug or _variant_slug(variant.task, variant.args)
        task_dir = out / slug
        (task_dir / "environment").mkdir(parents=True, exist_ok=True)
        (task_dir / "tests").mkdir(parents=True, exist_ok=True)

        prompt = await _materialize_prompt(env, variant.task, variant.args)
        (task_dir / "instruction.md").write_text(prompt, encoding="utf-8")

        task_toml = (
            f'id = "{slug}"\n'
            f'task = "{variant.task}"\n'
            f"args = {json.dumps(variant.args)}\n"
        )
        (task_dir / "task.toml").write_text(task_toml, encoding="utf-8")

        if dockerfile is not None:
            shutil.copyfile(dockerfile, task_dir / "environment" / "Dockerfile")

        test_sh = _TEST_SH.format(task=variant.task, args_json=json.dumps(variant.args))
        (task_dir / "tests" / "test.sh").write_text(test_sh, encoding="utf-8")

        created.append(task_dir)

    return created


__all__ = ["ALLOWED_PROTOCOLS", "export"]
