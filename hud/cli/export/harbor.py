"""HUD → Harbor exporter.

Produces a Harbor-format directory tree from a HUD taskset+environment.

Output layout (per ../outline-harbor-coding):
    <root>/
    ├── README.md
    ├── sample-run.sh
    ├── tasks/
    │   └── <task-slug>/
    │       ├── instruction.md            # rendered system + user prompt
    │       ├── task.toml                 # metadata, timeouts, resources
    │       ├── environment/
    │       │   ├── Dockerfile            # FROM <base> + bake-setup
    │       │   └── bake-setup.sh
    │       └── tests/
    │           └── test.sh               # generic; calls `hud scenario grade`
    └── manifest.json                     # taskset metadata + sample command

Setup is baked into each per-task image via `hud scenario setup` at
docker-build time (see bake-setup.sh). Grading shells to `hud scenario
grade` at runtime (see test.sh). One generic test.sh handles every
HUD scenario type.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .base import BaseExporter, ExportInput, ExportResult

if TYPE_CHECKING:
    from hud.eval.task import Task

__all__ = ["HarborExporter"]

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _read_template(name: str) -> str:
    return (_TEMPLATES_DIR / name).read_text(encoding="utf-8")


def _slugify(value: str) -> str:
    """Normalize an arbitrary identifier to a Harbor-safe slug."""
    s = value.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-") or "task"


def _task_slug(task: Task, fallback_index: int) -> str:
    candidate = task.slug or task.id or task.scenario or f"task-{fallback_index}"
    return _slugify(candidate)


def _toml_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _render_keywords(keywords: list[Any]) -> str:
    parts = [f'"{_toml_escape(str(k))}"' for k in keywords]
    return "[" + ", ".join(parts) + "]"


def _system_prompt(task: Task) -> str | None:
    cfg = task.agent_config
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        prompt = cfg.get("system_prompt")
    else:
        prompt = getattr(cfg, "system_prompt", None)
    return prompt or None


def _agent_timeout(task: Task) -> int:
    cfg = task.agent_config
    val = cfg.get("timeout") if isinstance(cfg, dict) else getattr(cfg, "timeout", None)
    try:
        return int(val) if val else 1800
    except (TypeError, ValueError):
        return 1800


def _build_instruction_md(task: Task, rendered_prompt: str) -> str:
    system = _system_prompt(task)
    if system:
        return f"{system.strip()}\n\n---\n\n{rendered_prompt.strip()}\n"
    return f"{rendered_prompt.strip()}\n"


def _derive_description(task: Task, slug: str, instruction: str, prompt_was_rendered: bool) -> str:
    """Pick a short, human-readable description for task.toml.

    Prefers the first line of a rendered instruction; falls back to
    args.description (a common convention) and finally the slug.
    """
    if prompt_was_rendered:
        first_line = next((ln for ln in instruction.splitlines() if ln.strip()), "")
        if first_line:
            return first_line[:200]
    args = task.args or {}
    if isinstance(args, dict):
        for key in ("description", "task_description", "prompt"):
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().splitlines()[0][:200]
    return slug


def _build_task_toml(
    task: Task,
    slug: str,
    taskset_name: str,
    instruction: str,
    prompt_was_rendered: bool,
) -> str:
    name = f"{_slugify(taskset_name)}/{slug}"
    description_seed = _derive_description(task, slug, instruction, prompt_was_rendered)
    keywords = task.metadata.get("keywords") or []
    if not isinstance(keywords, list):
        keywords = [keywords]

    difficulty = task.metadata.get("difficulty")
    category = task.metadata.get("category")
    verifier_timeout = float(task.metadata.get("verifier_timeout", 1800))
    agent_timeout = float(_agent_timeout(task))
    build_timeout = float(task.metadata.get("build_timeout", 2400))
    cpus = int(task.metadata.get("cpus", 4))
    memory_mb = int(task.metadata.get("memory_mb", 16384))

    lines: list[str] = [
        "[task]",
        f'name = "{_toml_escape(name)}"',
        f'description = "{_toml_escape(description_seed)}"',
        f"keywords = {_render_keywords(keywords)}",
        "",
        "[metadata]",
    ]
    if difficulty:
        lines.append(f'difficulty = "{_toml_escape(str(difficulty))}"')
    if category:
        lines.append(f'category = "{_toml_escape(str(category))}"')
    lines.extend(
        [
            "",
            "[verifier]",
            f"timeout_sec = {verifier_timeout}",
            'user = "root"',
            "",
            "[agent]",
            f"timeout_sec = {agent_timeout}",
            "",
            "[environment]",
            f"build_timeout_sec = {build_timeout}",
            f"cpus = {cpus}",
            f"memory_mb = {memory_mb}",
            "",
            "[[environment.mcp_servers]]",
            'name = "hud_env"',
            'transport = "streamable-http"',
            'url = "http://localhost:8765/mcp"',
            "",
            "[environment.healthcheck]",
            # MCP at /mcp speaks streamable-http (needs POST + handshake),
            # so a plain ``curl -fsS GET /mcp`` 400s and curl exits 22.
            # Use a TCP-level connect check instead: any successful
            # connection means ``hud dev`` is bound and ready.
            'command = "python3 -c \\"import socket;socket.create_connection((\'localhost\',8765),timeout=2).close()\\""',  # noqa: E501
            "interval_sec = 2.0",
            "timeout_sec = 5.0",
            "start_period_sec = 5.0",
            "retries = 30",
            "",
        ]
    )
    return "\n".join(lines)


def _build_task_dockerfile(
    base_image: str,
    scenario_full: str,
    args_json: str,
    required_env: list[str],
    platform: str | None = None,
) -> str:
    safe_args = args_json.replace("'", "'\\''")
    from_line = f"FROM --platform={platform} {base_image}" if platform else f"FROM {base_image}"
    parts = [
        from_line,
        "",
        f"ENV HUD_TASK_SCENARIO={scenario_full}",
        f"ENV HUD_TASK_ARGS='{safe_args}'",
        "",
        "COPY bake-setup.sh /opt/hud/bake-setup.sh",
        "RUN chmod +x /opt/hud/bake-setup.sh && /opt/hud/bake-setup.sh",
        "",
        "COPY entrypoint.py /opt/hud/entrypoint.py",
        "RUN chmod +x /opt/hud/entrypoint.py",
    ]
    if required_env:
        parts.append("")
        parts.extend(f"# {var} must be supplied at run time" for var in sorted(required_env))
    parts.extend(
        [
            "",
            'ENTRYPOINT ["python3", "/opt/hud/entrypoint.py"]',
            'CMD ["sleep", "infinity"]',
            "",
        ]
    )
    return "\n".join(parts)


def _full_scenario_name(task: Task, taskset_name: str) -> str:
    """Compose env:scenario form for `hud scenario` lookups."""
    if not task.scenario:
        raise ValueError(f"Task {task.slug or task.id!r} has no scenario; cannot export")
    if ":" in task.scenario:
        return task.scenario
    env_name: str | None = None
    env = task.env
    if env is not None:
        env_name = getattr(env, "name", None)
        if env_name is None and isinstance(env, dict):
            env_name = env.get("name")
    if not env_name:
        env_name = _slugify(taskset_name)
    return f"{env_name}:{task.scenario}"


def _select_tasks(inp: ExportInput) -> list[Task]:
    if not inp.task_subset:
        return list(inp.tasks)
    wanted = set(inp.task_subset)
    return [
        t for t in inp.tasks if (t.slug in wanted) or (t.id in wanted) or (t.scenario in wanted)
    ]


def _render_readme(taskset_name: str, task_slugs: list[str], base_image: str) -> str:
    listing = "\n".join(f"- `tasks/{s}/`" for s in task_slugs)
    return (
        f"# {taskset_name} (Harbor export)\n"
        f"\n"
        f"Auto-generated by `hud export harbor`. Each task is a self-contained\n"
        f"Harbor task directory under `tasks/`.\n"
        f"\n"
        f"## Base image\n"
        f"\n"
        f"    {base_image}\n"
        f"\n"
        f"## Tasks ({len(task_slugs)})\n"
        f"\n"
        f"{listing}\n"
        f"\n"
        f"## Run a task\n"
        f"\n"
        f"    ./sample-run.sh <task-slug>\n"
        f"\n"
        f"Reward is written to `logs/<task-slug>/verifier/reward.txt`.\n"
    )


class HarborExporter(BaseExporter):
    name = "harbor"
    description = "Harbor framework task layout (instruction.md + task.toml + tests/test.sh)"

    def export(self, inp: ExportInput) -> ExportResult:
        tasks = _select_tasks(inp)
        if not tasks:
            raise ValueError("No tasks selected for export")

        files: dict[str, str | bytes] = {}
        summary: list[str] = []
        slugs_used: set[str] = set()
        ordered_slugs: list[str] = []

        bake_setup = _read_template("bake-setup.sh")
        entrypoint_py = _read_template("entrypoint.py")
        test_sh = _read_template("test.sh")

        for index, task in enumerate(tasks):
            slug = _task_slug(task, index)
            base_slug = slug
            dedupe = 1
            while slug in slugs_used:
                dedupe += 1
                slug = f"{base_slug}-{dedupe}"
            slugs_used.add(slug)
            ordered_slugs.append(slug)

            scenario_full = _full_scenario_name(task, inp.taskset_name)
            args_json = json.dumps(task.args or {}, sort_keys=True)
            prompt_was_rendered = bool(
                inp.rendered_prompts.get(slug) or inp.rendered_prompts.get(task.slug or "")
            )
            rendered = inp.rendered_prompts.get(slug) or inp.rendered_prompts.get(
                task.slug or "", ""
            )
            if not rendered:
                rendered = (
                    f"[Prompt for scenario {scenario_full!r} was not rendered "
                    f"during export. Run with --render-prompts live or trace.]"
                )

            instruction = _build_instruction_md(task, rendered)
            task_toml = _build_task_toml(
                task, slug, inp.taskset_name, instruction, prompt_was_rendered
            )
            dockerfile = _build_task_dockerfile(
                base_image=inp.env_image,
                scenario_full=scenario_full,
                args_json=args_json,
                required_env=inp.env_required_env,
                platform=inp.env_platform,
            )

            base = f"tasks/{slug}"
            files[f"{base}/instruction.md"] = instruction
            files[f"{base}/task.toml"] = task_toml
            files[f"{base}/environment/Dockerfile"] = dockerfile
            files[f"{base}/environment/bake-setup.sh"] = bake_setup
            files[f"{base}/environment/entrypoint.py"] = entrypoint_py
            files[f"{base}/tests/test.sh"] = test_sh

            summary.append(f"{slug} ({scenario_full})")

        readme = _render_readme(inp.taskset_name, ordered_slugs, inp.env_image)
        sample_run = _read_template("sample-run.sh").replace("__FIRST_TASK__", ordered_slugs[0])

        files["README.md"] = readme
        files["sample-run.sh"] = sample_run

        manifest: dict[str, Any] = {
            "format": "harbor",
            "taskset_name": inp.taskset_name,
            "taskset_id": inp.taskset_id,
            "task_count": len(ordered_slugs),
            "task_slugs": ordered_slugs,
            "base_image": inp.env_image,
            "required_env": sorted(inp.env_required_env),
            "sample_run_command": f"./sample-run.sh {ordered_slugs[0]}",
        }
        files["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True) + "\n"

        return ExportResult(
            files=files,
            manifest=manifest,
            sample_run_script=sample_run,
            summary=summary,
        )
