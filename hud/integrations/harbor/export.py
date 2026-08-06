"""Export HUD tasks to self-contained Harbor task folders."""

from __future__ import annotations

import json
import shlex
import shutil
from pathlib import Path
from typing import Any

from hud.environment import Environment, load_environment
from hud.environment.server import TaskRunner
from hud.eval import Taskset
from hud.utils.naming import normalize_environment_name

ALLOWED_PROTOCOLS = ("ssh", "mcp")
DEFAULT_ANSWER_FILE = "/workspace/answer.txt"
CONTROL_PORT = 8765
BUILD_CONTEXT_IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", ".git", ".venv", "venv", "*.egg-info", ".pytest_cache"
)

ENTRYPOINT_SH = """\
#!/bin/sh
set -u

hud serve {serve_target} --port {port} &

hud task start {task} --args {args_json} --url tcp://127.0.0.1:{port} || {{
    echo "hud: task setup failed; refusing to run the agent against an unset task" >&2
    exit 1
}}

exec "$@"
"""

TEST_SH = """\
#!/bin/sh
set -u
mkdir -p /logs/verifier

ANSWER_FILE={answer_file}
[ -f "$ANSWER_FILE" ] || : > "$ANSWER_FILE"

if ! hud task grade {task} --args {args_json} --answer-file "$ANSWER_FILE" \\
    --url tcp://127.0.0.1:{port} > /logs/verifier/reward.txt 2> /logs/verifier/grade.err; then
    rm -f /logs/verifier/reward.txt
    echo "hud: grading failed; see /logs/verifier/grade.err" >&2
    exit 1
fi
"""


async def export(
    source: str,
    out_dir: str | Path,
    *,
    answer_file: str = DEFAULT_ANSWER_FILE,
    timeout_sec: float = 600.0,
) -> list[Path]:
    """Export HUD tasks from *source* into Harbor task folders under *out_dir*.

    ``source`` is a Python task source or a JSON/JSONL taskset next to its
    authored environment and Dockerfile. Each task becomes one self-contained
    Harbor task folder.
    """
    src = Path(source).resolve()
    source_dir = src.parent if src.is_file() else src
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    tasks = list(Taskset.from_file(src))

    scan = source_dir if src.suffix in (".json", ".jsonl") else src
    authored = {
        name: load_environment(scan, name=name)
        for name in dict.fromkeys(task.env for task in tasks)
    }
    serve_source = src.name if src.suffix == ".py" else "."

    dockerfile = next(
        (
            source_dir / name
            for name in ("Dockerfile.hud", "Dockerfile")
            if (source_dir / name).is_file()
        ),
        None,
    )
    if dockerfile is None:
        raise FileNotFoundError(
            f"no Dockerfile(.hud) next to {source_dir}; harbor export needs the env's "
            "build context to rebuild the image under Harbor.",
        )

    out_resolved = out.resolve()

    def ignore_export(dirpath: str, names: list[str]) -> set[str]:
        ignored = set(BUILD_CONTEXT_IGNORE(dirpath, names))
        base = Path(dirpath)
        ignored.update(name for name in names if (base / name).resolve() == out_resolved)
        return ignored

    created: list[Path] = []
    claimed: dict[str, str] = {}
    started: list[Environment] = []
    try:
        for env in authored.values():
            started.append(env)
            await env.start()
            unsupported = [
                capability.protocol
                for capability in env.capabilities
                if capability.protocol.split("/", 1)[0] not in ALLOWED_PROTOCOLS
            ]
            if unsupported:
                raise ValueError(
                    f"env {env.name!r} declares non-Harbor capabilities {unsupported}; "
                    f"only {'/'.join(ALLOWED_PROTOCOLS)} are convertible.",
                )

        for task in tasks:
            env = authored[task.env]
            if task.id not in env.tasks:
                raise TypeError(
                    f"harbor export needs a local env defining task {task.id!r} "
                    f"(an env.py named {task.env!r} next to the tasks); none was found.",
                )

            declared = task.slug
            if not any(character.isalnum() for character in declared):
                raise ValueError(f"task slug {declared!r} does not form a usable directory name")
            slug = normalize_environment_name(
                declared.replace("/", "-").replace("\\", "-"),
                default="harbor",
            )
            if slug in claimed:
                raise ValueError(
                    f"task slugs {claimed[slug]!r} and {declared!r} both name the export "
                    f"directory {slug!r}; give them distinct slugs"
                )
            claimed[slug] = declared

            task_dir = out / slug
            tests_dir = task_dir / "tests"
            tests_dir.mkdir(parents=True, exist_ok=True)

            runner = TaskRunner(env.tasks[task.id], task.args)
            try:
                payload = await runner.start()
            finally:
                await runner.cancel()
            prompt: Any = payload.get("prompt")
            if not isinstance(prompt, str):
                prompt = json.dumps(prompt, indent=2, default=str)
            (task_dir / "instruction.md").write_text(
                prompt
                + f"\n\n---\nWhen you have finished, write your final answer to `{answer_file}`.\n",
                encoding="utf-8",
                newline="\n",
            )

            args_json = json.dumps(task.args)
            (task_dir / "task.toml").write_text(
                'version = "1.0"\n'
                f"name = {json.dumps(slug)}\n"
                "\n[metadata]\n"
                f"hud_task = {json.dumps(task.id)}\n"
                f"hud_args = {json.dumps(args_json)}\n"
                "\n[agent]\n"
                f"timeout_sec = {timeout_sec}\n"
                "\n[verifier]\n"
                f"timeout_sec = {timeout_sec}\n",
                encoding="utf-8",
                newline="\n",
            )

            env_out = task_dir / "environment"
            if env_out.exists():
                shutil.rmtree(env_out)
            shutil.copytree(source_dir, env_out, ignore=ignore_export, symlinks=True)

            if src.is_file() and src.suffix in (".json", ".jsonl"):
                copied_taskset = env_out / src.name
                if copied_taskset.is_file():
                    copied_taskset.unlink()
            copied_ignore = env_out / ".dockerignore"
            if copied_ignore.is_file():
                copied_ignore.write_text(
                    copied_ignore.read_text("utf-8") + "\n!hud_entrypoint.sh\n",
                    encoding="utf-8",
                    newline="\n",
                )
            for name in ("Dockerfile.hud", "dockerfile"):
                alternate = env_out / name
                if alternate.exists():
                    alternate.unlink()

            (env_out / "hud_entrypoint.sh").write_text(
                ENTRYPOINT_SH.format(
                    port=CONTROL_PORT,
                    serve_target=shlex.quote(f"{serve_source}:{env.name}"),
                    task=shlex.quote(task.id),
                    args_json=shlex.quote(args_json),
                ),
                encoding="utf-8",
                newline="\n",
            )
            (env_out / "Dockerfile").write_text(
                dockerfile.read_text("utf-8").rstrip()
                + "\n\n"
                + "# HUD runtime for Harbor; final startup directives override the source image.\n"
                + "COPY --chmod=0755 hud_entrypoint.sh /hud_entrypoint.sh\n"
                + 'ENTRYPOINT ["/hud_entrypoint.sh"]\n'
                + 'CMD ["sh", "-c", "sleep infinity"]\n',
                encoding="utf-8",
                newline="\n",
            )
            (tests_dir / "test.sh").write_text(
                TEST_SH.format(
                    port=CONTROL_PORT,
                    task=shlex.quote(task.id),
                    args_json=shlex.quote(args_json),
                    answer_file=shlex.quote(answer_file),
                ),
                encoding="utf-8",
                newline="\n",
            )
            created.append(task_dir)
    finally:
        for env in reversed(started):
            await env.stop()

    return created
