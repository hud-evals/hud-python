"""Shared fixtures for Harbor converter tests.

Provides builders that create synthetic Harbor-format task directories
matching the terminal-bench-2 layout:

    task_name/
    ├── task.toml
    ├── instruction.md
    ├── environment/
    │   └── Dockerfile
    ├── tests/
    │   └── test.sh
    └── solution/          # optional, should be ignored by converter
"""

from __future__ import annotations

import textwrap
from pathlib import Path  # noqa: TC003 - used at runtime

import pytest

# ---------------------------------------------------------------------------
# task.toml templates (matching real terminal-bench style)
# ---------------------------------------------------------------------------

_DEFAULT_TASK_TOML = textwrap.dedent("""\
    [metadata]
    category = "systems"
    difficulty = "medium"
    tags = ["bash", "linux"]

    [verifier]
    timeout_sec = 120
""")

_TASK_TOML_WITH_IMAGE = textwrap.dedent("""\
    [metadata]
    category = "machine-learning"
    difficulty = "hard"
    tags = ["python", "ml"]

    [docker]
    image = "alexgshaw/caffe-cifar-10:20251031"

    [verifier]
    timeout_sec = 300
""")


# ---------------------------------------------------------------------------
# Dockerfile templates
# ---------------------------------------------------------------------------

_SIMPLE_DOCKERFILE = textwrap.dedent("""\
    FROM python:3.11-slim
    RUN apt-get update && apt-get install -y curl git
    WORKDIR /workspace
    CMD ["bash"]
""")

_ML_DOCKERFILE = textwrap.dedent("""\
    FROM nvidia/cuda:12.0-runtime-ubuntu22.04
    RUN apt-get update && apt-get install -y python3 python3-pip
    RUN pip3 install torch numpy
    WORKDIR /workspace
    ENTRYPOINT ["/bin/bash"]
""")


# ---------------------------------------------------------------------------
# Helper to build a single task directory
# ---------------------------------------------------------------------------


def make_harbor_task(
    parent: Path,
    name: str,
    instruction: str = "Solve the task.",
    task_toml: str = _DEFAULT_TASK_TOML,
    dockerfile: str | None = _SIMPLE_DOCKERFILE,
    test_script: str = '#!/bin/bash\necho "1.0" > /logs/verifier/reward.txt\n',
    include_solution: bool = False,
) -> Path:
    """Create a synthetic Harbor task directory under *parent*.

    Returns the task directory path.
    """
    task_dir = parent / name
    task_dir.mkdir(parents=True, exist_ok=True)

    (task_dir / "instruction.md").write_text(instruction, encoding="utf-8")
    (task_dir / "task.toml").write_text(task_toml, encoding="utf-8")

    if dockerfile is not None:
        env_dir = task_dir / "environment"
        env_dir.mkdir(exist_ok=True)
        (env_dir / "Dockerfile").write_text(dockerfile, encoding="utf-8")

    tests_dir = task_dir / "tests"
    tests_dir.mkdir(exist_ok=True)
    (tests_dir / "test.sh").write_text(test_script, encoding="utf-8")

    if include_solution:
        sol_dir = task_dir / "solution"
        sol_dir.mkdir(exist_ok=True)
        (sol_dir / "solve.sh").write_text("#!/bin/bash\necho done\n", encoding="utf-8")

    return task_dir


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def single_task(tmp_path: Path) -> Path:
    """A single Harbor task directory (like a standalone task)."""
    return make_harbor_task(
        tmp_path,
        "cancel-async-tasks",
        instruction=(
            "# Cancel Async Tasks\n\n"
            "Write a Python script that launches 5 asyncio tasks and cancels "
            "all of them within 2 seconds.\n"
        ),
    )


@pytest.fixture()
def dataset_same_env(tmp_path: Path) -> Path:
    """A dataset directory with 3 tasks sharing the same Dockerfile."""
    dataset = tmp_path / "terminal-bench-sample"
    dataset.mkdir()

    for name in ("cancel-async-tasks", "build-pmars", "chess-best-move"):
        make_harbor_task(
            dataset,
            name,
            instruction=f"# {name}\n\nSolve the {name} task.\n",
        )

    return dataset


@pytest.fixture()
def dataset_multi_env(tmp_path: Path) -> Path:
    """A dataset directory with tasks split across 2 different Dockerfiles."""
    dataset = tmp_path / "mixed-bench"
    dataset.mkdir()

    # Group 1: simple python tasks (same Dockerfile)
    for name in ("cancel-async-tasks", "build-pmars"):
        make_harbor_task(
            dataset,
            name,
            instruction=f"# {name}\n\nDo the thing.\n",
            dockerfile=_SIMPLE_DOCKERFILE,
        )

    # Group 2: ML tasks (different Dockerfile)
    for name in ("caffe-cifar-10", "sam-cell-seg"):
        make_harbor_task(
            dataset,
            name,
            instruction=f"# {name}\n\nTrain the model.\n",
            task_toml=_TASK_TOML_WITH_IMAGE,
            dockerfile=_ML_DOCKERFILE,
        )

    return dataset


@pytest.fixture()
def dataset_no_dockerfile(tmp_path: Path) -> Path:
    """A dataset where tasks have no environment/Dockerfile."""
    dataset = tmp_path / "no-docker-bench"
    dataset.mkdir()

    for name in ("task-a", "task-b"):
        make_harbor_task(
            dataset,
            name,
            instruction=f"# {name}\n\nSimple task.\n",
            dockerfile=None,  # No Dockerfile
        )

    return dataset


@pytest.fixture()
def dataset_with_solutions(tmp_path: Path) -> Path:
    """A dataset where tasks include solution/ directories."""
    dataset = tmp_path / "solved-bench"
    dataset.mkdir()

    for name in ("task-x", "task-y"):
        make_harbor_task(
            dataset,
            name,
            instruction=f"# {name}\n\nSolve it.\n",
            include_solution=True,
        )

    return dataset


@pytest.fixture()
def task_with_build_context(tmp_path: Path) -> Path:
    """A single task whose environment/ dir has extra build context files.

    Mimics build-pmars which has warriors/*.red files that the
    Dockerfile COPYs into the image.
    """
    task_dir = tmp_path / "build-pmars"
    task_dir.mkdir()

    (task_dir / "instruction.md").write_text(
        "# Build pMARS\n\nBuild the pMARS simulator.\n", encoding="utf-8"
    )
    (task_dir / "task.toml").write_text(
        textwrap.dedent("""\
            [metadata]
            category = "software-engineering"
            difficulty = "medium"

            [verifier]
            timeout_sec = 900
        """),
        encoding="utf-8",
    )

    # environment/ with Dockerfile AND extra build context files
    env_dir = task_dir / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text(
        textwrap.dedent("""\
            FROM debian:13.0-slim
            RUN apt-get update && apt-get install -y tmux
            WORKDIR /app
            COPY warriors/flashpaper.red warriors/rave.red /app/
        """),
        encoding="utf-8",
    )
    warriors = env_dir / "warriors"
    warriors.mkdir()
    (warriors / "flashpaper.red").write_text(";redcode\nMOV 0, 1\n", encoding="utf-8")
    (warriors / "rave.red").write_text(";redcode\nSPL 0, 0\n", encoding="utf-8")

    # tests/
    tests_dir = task_dir / "tests"
    tests_dir.mkdir()
    (tests_dir / "test.sh").write_text(
        '#!/bin/bash\necho "1.0" > /logs/verifier/reward.txt\n', encoding="utf-8"
    )

    return task_dir
