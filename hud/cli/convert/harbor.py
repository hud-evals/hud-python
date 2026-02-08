"""Harbor → HUD converter.

Converts Harbor framework tasks (task.toml + instruction.md + environment/ + tests/)
into HUD environments with scenarios and tasksets.

Harbor task structure:
    task_name/
    ├── instruction.md          # Agent prompt
    ├── task.toml               # Config: timeouts, metadata, resources
    ├── environment/
    │   └── Dockerfile          # Container the agent runs in
    ├── tests/
    │   └── test.sh             # Verification → writes reward.txt
    └── solution/               # Optional (ignored)

HUD output:
    hud-harbor-{dataset}/
    ├── env.py                  # Environment with run-task scenario
    ├── Dockerfile.hud          # Harbor Dockerfile + HUD MCP layer
    ├── pyproject.toml
    └── tasks/                  # All task data baked into image
        ├── task-a/
        │   ├── instruction.md
        │   └── tests/test.sh
        └── task-b/
            ├── instruction.md
            └── tests/test.sh
    taskset.json                # v5 taskset referencing the env
"""

from __future__ import annotations

import hashlib
import logging
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import BaseConverter, ConvertResult, GeneratedEnvironment

__all__ = ["HarborConverter"]

LOGGER = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================


def _is_harbor_task(path: Path) -> bool:
    """Check if a directory looks like a valid Harbor task."""
    return (
        path.is_dir()
        and (path / "task.toml").exists()
        and (path / "instruction.md").exists()
    )


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


def _normalize_name(name: str) -> str:
    """Normalize a dataset name to a valid HUD environment name."""
    normalized = name.strip().lower()
    normalized = normalized.replace(" ", "-").replace("_", "-")
    normalized = re.sub(r"[^a-z0-9-]", "", normalized)
    normalized = re.sub(r"-+", "-", normalized)
    return normalized.strip("-") or "converted"


def _find_dockerfile(env_dir: Path) -> str | None:
    """Read the Dockerfile from a Harbor environment directory."""
    for name in ("Dockerfile", "dockerfile"):
        path = env_dir / name
        if path.exists():
            return path.read_text(encoding="utf-8")
    return None


def _adapt_harbor_dockerfile(content: str) -> str:
    """Comment out CMD/ENTRYPOINT lines from a Harbor Dockerfile.

    These are replaced by the HUD MCP server entrypoint.
    """
    lines = content.splitlines()
    adapted: list[str] = []
    for line in lines:
        stripped = line.strip().upper()
        if stripped.startswith(("CMD ", "CMD[", "ENTRYPOINT ", "ENTRYPOINT[")):
            adapted.append(f"# [harbor original] {line}")
        else:
            adapted.append(line)
    return "\n".join(adapted)


# =============================================================================
# Data classes
# =============================================================================


@dataclass
class HarborTask:
    """Parsed Harbor task."""

    task_id: str
    directory: Path
    instruction: str
    config: dict[str, Any]
    env_hash: str


def _parse_task(task_dir: Path) -> HarborTask | None:
    """Parse a Harbor task directory into a HarborTask."""
    try:
        instruction = (task_dir / "instruction.md").read_text(encoding="utf-8")
    except Exception:
        LOGGER.warning("Failed to read instruction.md in %s", task_dir)
        return None

    try:
        raw = (task_dir / "task.toml").read_text(encoding="utf-8")
        config: dict[str, Any] = tomllib.loads(raw)
    except Exception:
        LOGGER.warning("Failed to parse task.toml in %s", task_dir)
        config = {}

    env_dir = task_dir / "environment"
    env_hash = _hash_directory(env_dir) if env_dir.exists() else "no-env"

    return HarborTask(
        task_id=task_dir.name,
        directory=task_dir,
        instruction=instruction,
        config=config,
        env_hash=env_hash,
    )


# =============================================================================
# Templates
# =============================================================================

# fmt: off

# Header + shared body split so the scenario signature can vary.
_ENV_PY_HEADER = '''\
"""{env_name} - HUD environment converted from Harbor.

Source: {source_path}
Tasks: {task_count}

This environment runs Harbor-format tasks. Each task has:
- instruction.md: the agent prompt
- tests/test.sh: verification script that writes reward to /logs/verifier/

The run-task scenario reads the instruction, lets the agent work,
then executes the test script and parses the reward.
"""

import json
import logging
import subprocess
from pathlib import Path
{extra_imports}
from hud import Environment
from hud.tools import BashTool, EditTool
from hud.tools.filesystem import GlobTool, GrepTool, ListTool, ReadTool

LOGGER = logging.getLogger(__name__)

TASKS_DIR = Path("/harbor/tasks")

env = Environment("{env_name}")

# Standard coding tools - agents interact via bash (matching Harbor's model)
env.add_tool(BashTool())
env.add_tool(EditTool())
env.add_tool(ReadTool())
env.add_tool(GrepTool())
env.add_tool(GlobTool())
env.add_tool(ListTool())

'''

# Single task: task_id is optional, defaults to the only task.
_SCENARIO_SINGLE = '''\
@env.scenario("run-task")
async def run_task(task_id: str = "{default_task_id}"):
'''

# Multiple tasks: task_id is required, typed as a Literal.
_SCENARIO_MULTI = '''\
TaskId = Literal[{task_id_literal}]


@env.scenario("run-task")
async def run_task(task_id: TaskId):
'''

_SCENARIO_BODY = '''\
    """Run a Harbor task by ID.

    Reads /harbor/tasks/<task_id>/instruction.md as the prompt.
    After the agent works, runs tests/test.sh and parses
    /logs/verifier/reward.txt or reward.json for the reward.
    """
    task_dir = TASKS_DIR / str(task_id)
    if not task_dir.exists():
        available = [d.name for d in TASKS_DIR.iterdir() if d.is_dir()]
        raise ValueError(
            f"Task '{{task_id}}' not found. Available: {{available}}"
        )

    # Read the task instruction
    instruction = (task_dir / "instruction.md").read_text(encoding="utf-8")

    # Setup: yield prompt to the agent
    answer = yield instruction

    # Ensure log output directory exists
    logs_dir = Path("/logs/verifier")
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Harbor mounts the task's tests/ directory at /tests/ — replicate that
    tests_link = Path("/tests")
    task_tests = task_dir / "tests"
    if task_tests.is_dir():
        if tests_link.is_symlink() or tests_link.exists():
            tests_link.unlink()
        tests_link.symlink_to(task_tests)

    # Evaluate: run the test script
    test_script = task_dir / "tests" / "test.sh"
    if test_script.exists():
        try:
            result = subprocess.run(
                ["bash", str(test_script)],
                cwd="/app",
                capture_output=True,
                text=True,
                timeout={verifier_timeout},
                check=False,
            )
            if result.stdout:
                LOGGER.info("test.sh stdout for %s:\\n%s", task_id, result.stdout[-2000:])
            if result.stderr:
                LOGGER.info("test.sh stderr for %s:\\n%s", task_id, result.stderr[-2000:])
            if result.returncode != 0:
                LOGGER.warning(
                    "test.sh exited with code %d for task %s",
                    result.returncode, task_id,
                )
        except subprocess.TimeoutExpired:
            LOGGER.warning("Test script timed out for task %s", task_id)
        except Exception as exc:
            LOGGER.warning("Test script failed for task %s: %s", task_id, exc)
    else:
        LOGGER.warning("No test script found at %s", test_script)

    # Parse and yield reward
    yield _parse_harbor_reward()


def _parse_harbor_reward() -> float:
    """Parse reward from Harbor standard output locations.

    Harbor test scripts write results to /logs/verifier/ as either:
    - reward.txt: a single float value
    - reward.json: {{"reward": float}} or just a float
    """
    reward_txt = Path("/logs/verifier/reward.txt")
    reward_json = Path("/logs/verifier/reward.json")

    if reward_txt.exists():
        try:
            return float(reward_txt.read_text(encoding="utf-8").strip())
        except ValueError:
            pass

    if reward_json.exists():
        try:
            data = json.loads(reward_json.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return float(data.get("reward", 0.0))
            return float(data)
        except (ValueError, json.JSONDecodeError):
            pass

    return 0.0
'''


def _build_env_py(
    env_name: str,
    source_path: str,
    task_ids: list[str],
    verifier_timeout: int,
) -> str:
    """Build the env.py content, adapting the scenario signature to task count."""
    if len(task_ids) == 1:
        extra_imports = ""
        scenario = _SCENARIO_SINGLE.format(default_task_id=task_ids[0])
    else:
        extra_imports = "\nfrom typing import Literal\n"
        literal_values = ", ".join(f'"{tid}"' for tid in sorted(task_ids))
        scenario = _SCENARIO_MULTI.format(task_id_literal=literal_values)

    header = _ENV_PY_HEADER.format(
        env_name=env_name,
        source_path=source_path,
        task_count=len(task_ids),
        extra_imports=extra_imports,
    )
    body = _SCENARIO_BODY.format(verifier_timeout=verifier_timeout)
    return header + scenario + body

# fmt: on

# Shared snippet: install uv standalone (works on any base image with curl or
# apt), then use uv to bootstrap Python and sync dependencies.
_HUD_LAYER = """\
# ============================================================
# HUD MCP server layer
# ============================================================
WORKDIR /hud

# Install uv standalone (no pip/python required on the base image)
RUN command -v curl >/dev/null 2>&1 || \\
    (apt-get update -qq && apt-get install -y -qq --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*) && \\
    curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev --no-install-project 2>/dev/null || uv sync --no-dev --no-install-project

# Harbor task data (instructions + test scripts baked into image)
COPY tasks/ /harbor/tasks/

# Ensure standard directories exist
RUN mkdir -p /logs/verifier /workspace

COPY env.py ./

CMD ["uv", "run", "--no-project", "python", "-m", "hud", "dev", "env:env", "--stdio"]
"""

DOCKERFILE_WITH_BASE_TEMPLATE = """\
# ============================================================
# Harbor environment base
# Source: {source}
# ============================================================
{base_dockerfile}
""" + _HUD_LAYER

DOCKERFILE_FALLBACK_TEMPLATE = """\
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \\
    curl git build-essential && rm -rf /var/lib/apt/lists/*
""" + _HUD_LAYER

PYPROJECT_TEMPLATE = """\
[project]
name = "{name}"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["hud-python", "openai"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
"""


# =============================================================================
# Converter
# =============================================================================


class HarborConverter(BaseConverter):
    """Convert Harbor tasks/datasets to HUD format.

    Handles:
    - Single task directory (has task.toml directly)
    - Dataset directory (subdirectories are Harbor tasks)
    - Multi-environment datasets (tasks grouped by Dockerfile hash)
    """

    name = "harbor"
    description = "Harbor framework (task.toml + instruction.md + environment/ + tests/)"

    def detect(self, path: Path) -> bool:
        if _is_harbor_task(path):
            return True
        # Check for dataset (directory containing task subdirectories)
        if path.is_dir():
            return any(_is_harbor_task(d) for d in path.iterdir() if d.is_dir())
        return False

    def convert(self, path: Path) -> ConvertResult:
        path = path.resolve()

        # Discover tasks
        if _is_harbor_task(path):
            task_dirs = [path]
            dataset_name = path.parent.name
        else:
            task_dirs = sorted(
                d for d in path.iterdir() if d.is_dir() and _is_harbor_task(d)
            )
            dataset_name = path.name

        if not task_dirs:
            raise ValueError(f"No Harbor tasks found in {path}")

        # Parse all tasks
        tasks: list[HarborTask] = []
        skipped = 0
        for td in task_dirs:
            parsed = _parse_task(td)
            if parsed:
                tasks.append(parsed)
            else:
                skipped += 1

        if not tasks:
            raise ValueError("All Harbor tasks failed to parse")

        if skipped:
            LOGGER.warning("Skipped %d task(s) that failed to parse", skipped)

        LOGGER.info("Parsed %d Harbor task(s) from %s", len(tasks), path)

        # Group by environment Dockerfile hash
        groups: dict[str, list[HarborTask]] = {}
        for task in tasks:
            groups.setdefault(task.env_hash, []).append(task)

        LOGGER.info("Found %d unique environment group(s)", len(groups))

        # Generate environments and taskset
        environments: list[GeneratedEnvironment] = []
        taskset: list[dict[str, Any]] = []
        base_name = f"hud-harbor-{_normalize_name(dataset_name)}"

        # Sort groups by size (largest first) for consistent naming
        sorted_groups = sorted(groups.items(), key=lambda x: -len(x[1]))

        for idx, (_env_hash, group_tasks) in enumerate(sorted_groups, start=1):
            # Naming: single group gets base_name, multiple get suffix
            env_name = base_name if len(sorted_groups) == 1 else f"{base_name}-g{idx}"

            # Use representative task for shared config
            rep_task = group_tasks[0]
            env_dir = rep_task.directory / "environment"
            dockerfile_content = _find_dockerfile(env_dir) if env_dir.exists() else None

            # Extract verifier timeout from config
            verifier_timeout = 600
            verifier_cfg = rep_task.config.get("verifier", {})
            if isinstance(verifier_cfg, dict):
                timeout_val = verifier_cfg.get("timeout_sec")
                if timeout_val is not None:
                    verifier_timeout = int(timeout_val)

            # --- Generate env.py ---
            # Use forward slashes in source_path to avoid unicode escape issues on Windows
            task_ids = [t.task_id for t in group_tasks]
            env_py = _build_env_py(
                env_name=env_name,
                source_path=path.as_posix(),
                task_ids=task_ids,
                verifier_timeout=verifier_timeout,
            )

            # --- Generate Dockerfile.hud ---
            if dockerfile_content:
                adapted = _adapt_harbor_dockerfile(dockerfile_content)
                dockerfile = DOCKERFILE_WITH_BASE_TEMPLATE.format(
                    source=env_dir.as_posix(),
                    base_dockerfile=adapted,
                )
            else:
                dockerfile = DOCKERFILE_FALLBACK_TEMPLATE

            # --- Generate pyproject.toml ---
            pyproject = PYPROJECT_TEMPLATE.format(name=env_name)

            # --- Map task IDs to source directories ---
            task_dir_map = {t.task_id: t.directory for t in group_tasks}

            # Build context: non-Dockerfile files from environment/ dir
            # (e.g., warriors/*.red that the Dockerfile COPYs)
            build_ctx = env_dir if env_dir.exists() else None

            environments.append(
                GeneratedEnvironment(
                    name=env_name,
                    env_py=env_py,
                    dockerfile=dockerfile,
                    pyproject_toml=pyproject,
                    task_dirs=task_dir_map,
                    build_context_source=build_ctx,
                )
            )

            # --- Generate v5 taskset entries ---
            for task in group_tasks:
                metadata: dict[str, Any] = {
                    "harbor_source": task.directory.relative_to(path.parent).as_posix(),
                }
                # Pull metadata from task.toml [metadata] section
                toml_meta = task.config.get("metadata", {})
                if isinstance(toml_meta, dict):
                    metadata.update(toml_meta)

                taskset.append(
                    {
                        "env": {"name": env_name},
                        "scenario": f"{env_name}:run-task",
                        "args": {"task_id": task.task_id},
                        "metadata": metadata,
                    }
                )

        # Build summary lines
        summary = [
            f"Converted {len(tasks)} Harbor task(s) into {len(environments)} environment(s).",
        ]
        if skipped:
            summary.append(f"Skipped {skipped} task(s) that failed to parse.")
        summary.append("")
        for env_gen in environments:
            task_count = len(env_gen.task_dirs)
            summary.append(f"  {env_gen.name}/ ({task_count} tasks)")
        summary.extend(
            [
                "",
                "Next steps:",
                "  1. hud deploy <environment>/",
                "  2. hud eval taskset.json",
            ]
        )

        return ConvertResult(
            environments=environments,
            taskset=taskset,
            summary=summary,
        )
