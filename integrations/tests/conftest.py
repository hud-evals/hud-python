"""Builders for synthetic Harbor-format task directories (terminal-bench layout):

task_name/
├── task.toml
├── instruction.md
├── environment/Dockerfile
└── tests/test.sh
"""

from __future__ import annotations

import textwrap
from pathlib import Path  # noqa: TC003 - used at runtime

import pytest

_DEFAULT_TASK_TOML = textwrap.dedent("""\
    [metadata]
    category = "systems"
    difficulty = "medium"
    tags = ["bash", "linux"]

    [verifier]
    timeout_sec = 120
""")

_ML_TASK_TOML = textwrap.dedent("""\
    [metadata]
    category = "machine-learning"
    difficulty = "hard"
    tags = ["python", "ml"]

    [docker]
    image = "alexgshaw/caffe-cifar-10:20251031"

    [verifier]
    timeout_sec = 300
""")

_SIMPLE_DOCKERFILE = textwrap.dedent("""\
    FROM python:3.11-slim
    RUN apt-get update && apt-get install -y curl git
    WORKDIR /workspace
    CMD ["bash"]
""")

_ML_DOCKERFILE = textwrap.dedent("""\
    FROM nvidia/cuda:12.0-runtime-ubuntu22.04
    RUN apt-get update && apt-get install -y python3 python3-pip
    WORKDIR /workspace
    ENTRYPOINT ["/bin/bash"]
""")


def make_harbor_task(
    parent: Path,
    name: str,
    instruction: str = "Solve the task.",
    task_toml: str = _DEFAULT_TASK_TOML,
    dockerfile: str | None = _SIMPLE_DOCKERFILE,
) -> Path:
    """Create a synthetic Harbor task directory under *parent*; return it."""
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
    (tests_dir / "test.sh").write_text(
        '#!/bin/bash\necho "1.0" > /logs/verifier/reward.txt\n', encoding="utf-8"
    )
    return task_dir


@pytest.fixture()
def single_task(tmp_path: Path) -> Path:
    """A single standalone Harbor task directory."""
    return make_harbor_task(
        tmp_path,
        "cancel-async-tasks",
        instruction="# Cancel Async Tasks\n\nCancel 5 asyncio tasks within 2 seconds.\n",
    )


@pytest.fixture()
def dataset_same_env(tmp_path: Path) -> Path:
    """A dataset directory with 3 tasks sharing the same Dockerfile."""
    dataset = tmp_path / "terminal-bench-sample"
    dataset.mkdir()
    for name in ("cancel-async-tasks", "build-pmars", "chess-best-move"):
        make_harbor_task(dataset, name, instruction=f"# {name}\n\nSolve the {name} task.\n")
    return dataset


@pytest.fixture()
def dataset_multi_env(tmp_path: Path) -> Path:
    """A dataset directory with tasks split across 2 different Dockerfiles."""
    dataset = tmp_path / "Mixed Bench"
    dataset.mkdir()
    for name in ("cancel-async-tasks", "build-pmars"):
        make_harbor_task(dataset, name, dockerfile=_SIMPLE_DOCKERFILE)
    for name in ("caffe-cifar-10", "sam-cell-seg"):
        make_harbor_task(dataset, name, task_toml=_ML_TASK_TOML, dockerfile=_ML_DOCKERFILE)
    return dataset
