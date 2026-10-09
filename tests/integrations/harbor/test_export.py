"""``harbor.export`` writes Harbor task folders whose scripts drive the ``hud`` CLI.

The default lane snapshots what export writes and runs the generated
entrypoint with a stub ``hud`` on ``PATH``, recording the commands it issues.
The docker lane builds the exported image, starts it the way Harbor does, and
runs ``tests/test.sh`` inside it.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import textwrap
import tomllib
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from inline_snapshot import snapshot

from hud.integrations.harbor import export

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.harness import HudEnv

REPO = Path(__file__).resolve().parents[3]

ENV_PY = """\
import os
from pathlib import Path

from hud import Environment

env = Environment("demo")
markers = os.environ.get("EXPORT_TEST_MARKERS")


@env.initialize
async def started() -> None:
    if markers:
        Path(markers, "initialized").touch()


@env.shutdown
async def stopped() -> None:
    if markers:
        Path(markers, "stopped").touch()


@env.template()
async def solve(n: int = 1, prompt: str = ""):
    answer = yield f"solve {n}: {prompt}"
    if answer == "crash":
        raise RuntimeError("the grader is broken")
    yield 1.0 if answer == str(n) else 0.25


tasks = [solve(n=2, prompt="what's next?")]
"""

DOCKERFILE = 'FROM python:3.12-slim\nRUN pip install hud\nCMD ["hud", "serve", "env:env"]\n'

STUB_HUD = """\
#!{python}
import json, os, sys

with open(os.environ["STUB_HUD_LOG"], "a") as log:
    log.write(json.dumps(sys.argv[1:]) + "\\n")
if sys.argv[1:3] == ["task", "start"]:
    sys.exit(int(os.environ.get("STUB_HUD_START_EXIT", "0")))
"""


def write_source(root: Path, files: dict[str, str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for relative, content in {"env.py": ENV_PY, "Dockerfile": DOCKERFILE, **files}.items():
        (root / relative).write_text(content, encoding="utf-8")
    return root


def written(task_dir: Path) -> dict[str, Any]:
    files = sorted(
        path.relative_to(task_dir).as_posix() for path in task_dir.rglob("*") if path.is_file()
    )
    environment = task_dir / "environment"
    ignore = environment / ".dockerignore"
    return {
        "files": files,
        "task.toml": tomllib.loads((task_dir / "task.toml").read_text("utf-8")),
        "instruction.md": (task_dir / "instruction.md").read_text("utf-8"),
        "Dockerfile": (environment / "Dockerfile").read_text("utf-8"),
        **({".dockerignore": ignore.read_text("utf-8")} if ignore.is_file() else {}),
    }


def names(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir())


def run_entrypoint(task_dir: Path, tmp_path: Path, *, start_exit: int = 0) -> dict[str, Any]:
    """Run ``hud_entrypoint.sh`` with a stub ``hud`` and a command that leaves a marker."""
    stub_bin = tmp_path / "stub-bin"
    stub_bin.mkdir(exist_ok=True)
    stub = stub_bin / "hud"
    stub.write_text(STUB_HUD.format(python=sys.executable), encoding="utf-8")
    stub.chmod(0o755)
    log = tmp_path / f"hud-{uuid.uuid4().hex}.jsonl"
    marker = tmp_path / "agent-ran"
    marker.unlink(missing_ok=True)
    completed = subprocess.run(
        ["sh", str(task_dir / "environment/hud_entrypoint.sh"), "touch", str(marker)],
        env={
            **os.environ,
            "PATH": f"{stub_bin}{os.pathsep}{os.environ['PATH']}",
            "STUB_HUD_LOG": str(log),
            "STUB_HUD_START_EXIT": str(start_exit),
        },
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return {
        "exit": completed.returncode,
        "stderr": completed.stderr,
        "agent ran": marker.exists(),
        # ``hud serve`` runs in the background, so the two calls race.
        "hud calls": sorted(json.loads(line) for line in log.read_text("utf-8").splitlines()),
    }


def main_thread_guard(source: Path) -> None:
    env_py = source / "env.py"
    env_py.write_text(
        "import threading\nassert threading.current_thread() is threading.main_thread()\n"
        + env_py.read_text("utf-8"),
        encoding="utf-8",
    )


def rows(*entries: dict[str, Any]) -> Callable[[Path], None]:
    def write(source: Path) -> None:
        suffix = ".jsonl" if len(entries) > 1 else ".json"
        lines = [json.dumps(entry) for entry in entries]
        (source / f"tasks{suffix}").write_text(
            "\n".join(lines) if suffix == ".jsonl" else json.dumps(list(entries)),
            encoding="utf-8",
        )

    return write


@pytest.fixture
def markers(tmp_path: Path, hud_env: HudEnv) -> Path:
    """Where the source environment's initialize and shutdown hooks leave a file."""
    directory = tmp_path / "markers"
    directory.mkdir()
    hud_env.set(EXPORT_TEST_MARKERS=str(directory))
    return directory


async def test_export_writes_a_harbor_task_whose_entrypoint_parks_the_task(
    tmp_path: Path, markers: Path
) -> None:
    source = write_source(tmp_path / "source", {})

    (task_dir,) = await export(str(source / "env.py"), tmp_path / "out")

    assert {**written(task_dir), "entrypoint": run_entrypoint(task_dir, tmp_path)} == snapshot(
        {
            "files": [
                "environment/Dockerfile",
                "environment/env.py",
                "environment/hud_entrypoint.sh",
                "instruction.md",
                "task.toml",
                "tests/test.sh",
            ],
            "task.toml": {
                "version": "1.0",
                "name": "solve-4080368a",
                "metadata": {
                    "hud_task": "solve",
                    "hud_args": '{"n": 2, "prompt": "what\'s next?"}',
                },
                "agent": {"timeout_sec": 600.0},
                "verifier": {"timeout_sec": 600.0},
            },
            "instruction.md": """\
solve 2: what's next?

---
When you have finished, write your final answer to `/workspace/answer.txt`.
""",
            "Dockerfile": """\
FROM python:3.12-slim
RUN pip install hud
CMD ["hud", "serve", "env:env"]

# HUD runtime for Harbor; final startup directives override the source image.
COPY --chmod=0755 hud_entrypoint.sh /hud_entrypoint.sh
ENTRYPOINT ["/hud_entrypoint.sh"]
CMD ["sh", "-c", "sleep infinity"]
""",
            "entrypoint": {
                "exit": 0,
                "stderr": "",
                "agent ran": True,
                "hud calls": [
                    ["serve", "env.py:demo", "--port", "8765"],
                    [
                        "task",
                        "start",
                        "solve",
                        "--args",
                        '{"n": 2, "prompt": "what\'s next?"}',
                        "--url",
                        "tcp://127.0.0.1:8765",
                    ],
                ],
            },
        }
    )
    assert names(markers) == ["initialized", "stopped"]


SOURCES: dict[str, dict[str, Any]] = {
    "a python source importing on the main thread": {"source": "env.py", "edit": main_thread_guard},
    "an environment bound under another name": {
        "source": "tasks.py",
        "files": {
            "tasks.py": ENV_PY.replace("env = Environment", "bench = Environment").replace(
                "@env.", "@bench."
            )
        },
    },
    "a json taskset with a nested slug": {
        "source": "tasks.json",
        "edit": rows({"env": "demo", "id": "solve", "args": {"n": 3}, "slug": "suite/fix"}),
    },
    "a jsonl taskset": {
        "source": "tasks.jsonl",
        "edit": rows(
            {"env": "demo", "id": "solve", "args": {"n": 4}, "slug": "four"},
            {"env": "demo", "id": "solve", "args": {"n": 5}, "slug": "five"},
        ),
    },
    "a hud dockerfile beside the plain one": {
        "source": "env.py",
        "files": {"Dockerfile.hud": "FROM python:3.12-slim AS hud\nRUN pip install hud\n"},
        "show": ["Dockerfile"],
    },
    "a restrictive dockerignore and build files": {
        "source": "env.py",
        "files": {".dockerignore": "*\n", "package.json": '{"name": "app"}\n'},
        "show": [".dockerignore"],
    },
    "a non-root multiline image": {
        "source": "env.py",
        "files": {
            "Dockerfile": (
                "FROM python:3.12-slim\nRUN useradd -m app\nUSER app\n"
                'CMD ["python", \\\n    "-m", \\\n    "app"]\n'
            )
        },
        "show": ["Dockerfile"],
    },
    "a custom answer file": {
        "source": "env.py",
        "answer_file": "/app/out.txt",
        "show": ["instruction.md"],
    },
}


@pytest.mark.parametrize("case", list(SOURCES))
async def test_export_maps_each_source_form_to_harbor_task_folders(
    case: str, tmp_path: Path, markers: Path
) -> None:
    spec = SOURCES[case]
    source = write_source(tmp_path / "source", spec.get("files", {}))
    if edit := spec.get("edit"):
        edit(source)

    created = await export(
        str(source / spec["source"]),
        tmp_path / "out",
        answer_file=spec.get("answer_file", "/workspace/answer.txt"),
    )

    observed = {}
    for task_dir in created:
        files = written(task_dir)
        observed[task_dir.name] = {
            "files": files["files"],
            "hud calls": run_entrypoint(task_dir, tmp_path)["hud calls"],
            **{key: files[key] for key in spec.get("show", [])},
        }
    assert (
        observed
        == snapshot(
            {
                "a python source importing on the main thread": {
                    "solve-4080368a": {
                        "files": [
                            "environment/Dockerfile",
                            "environment/env.py",
                            "environment/hud_entrypoint.sh",
                            "instruction.md",
                            "task.toml",
                            "tests/test.sh",
                        ],
                        "hud calls": [
                            ["serve", "env.py:demo", "--port", "8765"],
                            [
                                "task",
                                "start",
                                "solve",
                                "--args",
                                '{"n": 2, "prompt": "what\'s next?"}',
                                "--url",
                                "tcp://127.0.0.1:8765",
                            ],
                        ],
                    }
                },
                "an environment bound under another name": {
                    "solve-4080368a": {
                        "files": [
                            "environment/Dockerfile",
                            "environment/env.py",
                            "environment/hud_entrypoint.sh",
                            "environment/tasks.py",
                            "instruction.md",
                            "task.toml",
                            "tests/test.sh",
                        ],
                        "hud calls": [
                            ["serve", "tasks.py:demo", "--port", "8765"],
                            [
                                "task",
                                "start",
                                "solve",
                                "--args",
                                '{"n": 2, "prompt": "what\'s next?"}',
                                "--url",
                                "tcp://127.0.0.1:8765",
                            ],
                        ],
                    }
                },
                "a json taskset with a nested slug": {
                    "suite-fix": {
                        "files": [
                            "environment/Dockerfile",
                            "environment/env.py",
                            "environment/hud_entrypoint.sh",
                            "instruction.md",
                            "task.toml",
                            "tests/test.sh",
                        ],
                        "hud calls": [
                            ["serve", ".:demo", "--port", "8765"],
                            [
                                "task",
                                "start",
                                "solve",
                                "--args",
                                '{"n": 3}',
                                "--url",
                                "tcp://127.0.0.1:8765",
                            ],
                        ],
                    }
                },
                "a jsonl taskset": {
                    "four": {
                        "files": [
                            "environment/Dockerfile",
                            "environment/env.py",
                            "environment/hud_entrypoint.sh",
                            "instruction.md",
                            "task.toml",
                            "tests/test.sh",
                        ],
                        "hud calls": [
                            ["serve", ".:demo", "--port", "8765"],
                            [
                                "task",
                                "start",
                                "solve",
                                "--args",
                                '{"n": 4}',
                                "--url",
                                "tcp://127.0.0.1:8765",
                            ],
                        ],
                    },
                    "five": {
                        "files": [
                            "environment/Dockerfile",
                            "environment/env.py",
                            "environment/hud_entrypoint.sh",
                            "instruction.md",
                            "task.toml",
                            "tests/test.sh",
                        ],
                        "hud calls": [
                            ["serve", ".:demo", "--port", "8765"],
                            [
                                "task",
                                "start",
                                "solve",
                                "--args",
                                '{"n": 5}',
                                "--url",
                                "tcp://127.0.0.1:8765",
                            ],
                        ],
                    },
                },
                "a hud dockerfile beside the plain one": {
                    "solve-4080368a": {
                        "files": [
                            "environment/Dockerfile",
                            "environment/env.py",
                            "environment/hud_entrypoint.sh",
                            "instruction.md",
                            "task.toml",
                            "tests/test.sh",
                        ],
                        "hud calls": [
                            ["serve", "env.py:demo", "--port", "8765"],
                            [
                                "task",
                                "start",
                                "solve",
                                "--args",
                                '{"n": 2, "prompt": "what\'s next?"}',
                                "--url",
                                "tcp://127.0.0.1:8765",
                            ],
                        ],
                        "Dockerfile": """\
FROM python:3.12-slim AS hud
RUN pip install hud

# HUD runtime for Harbor; final startup directives override the source image.
COPY --chmod=0755 hud_entrypoint.sh /hud_entrypoint.sh
ENTRYPOINT ["/hud_entrypoint.sh"]
CMD ["sh", "-c", "sleep infinity"]
""",
                    }
                },
                "a restrictive dockerignore and build files": {
                    "solve-4080368a": {
                        "files": [
                            "environment/.dockerignore",
                            "environment/Dockerfile",
                            "environment/env.py",
                            "environment/hud_entrypoint.sh",
                            "environment/package.json",
                            "instruction.md",
                            "task.toml",
                            "tests/test.sh",
                        ],
                        "hud calls": [
                            ["serve", "env.py:demo", "--port", "8765"],
                            [
                                "task",
                                "start",
                                "solve",
                                "--args",
                                '{"n": 2, "prompt": "what\'s next?"}',
                                "--url",
                                "tcp://127.0.0.1:8765",
                            ],
                        ],
                        ".dockerignore": """\
*

!hud_entrypoint.sh
""",
                    }
                },
                "a non-root multiline image": {
                    "solve-4080368a": {
                        "files": [
                            "environment/Dockerfile",
                            "environment/env.py",
                            "environment/hud_entrypoint.sh",
                            "instruction.md",
                            "task.toml",
                            "tests/test.sh",
                        ],
                        "hud calls": [
                            ["serve", "env.py:demo", "--port", "8765"],
                            [
                                "task",
                                "start",
                                "solve",
                                "--args",
                                '{"n": 2, "prompt": "what\'s next?"}',
                                "--url",
                                "tcp://127.0.0.1:8765",
                            ],
                        ],
                        "Dockerfile": """\
FROM python:3.12-slim
RUN useradd -m app
USER app
CMD ["python", \\
    "-m", \\
    "app"]

# HUD runtime for Harbor; final startup directives override the source image.
COPY --chmod=0755 hud_entrypoint.sh /hud_entrypoint.sh
ENTRYPOINT ["/hud_entrypoint.sh"]
CMD ["sh", "-c", "sleep infinity"]
""",
                    }
                },
                "a custom answer file": {
                    "solve-4080368a": {
                        "files": [
                            "environment/Dockerfile",
                            "environment/env.py",
                            "environment/hud_entrypoint.sh",
                            "instruction.md",
                            "task.toml",
                            "tests/test.sh",
                        ],
                        "hud calls": [
                            ["serve", "env.py:demo", "--port", "8765"],
                            [
                                "task",
                                "start",
                                "solve",
                                "--args",
                                '{"n": 2, "prompt": "what\'s next?"}',
                                "--url",
                                "tcp://127.0.0.1:8765",
                            ],
                        ],
                        "instruction.md": """\
solve 2: what's next?

---
When you have finished, write your final answer to `/app/out.txt`.
""",
                    }
                },
            }
        )[case]
    )
    assert names(markers) == ["initialized", "stopped"]


@pytest.mark.parametrize(
    ("case", "edit", "error", "message", "started"),
    [
        (
            "no dockerfile",
            lambda source: (source / "Dockerfile").unlink(),
            FileNotFoundError,
            "no Dockerfile",
            False,
        ),
        (
            "a capability harbor cannot carry",
            lambda source: (source / "env.py").write_text(
                ENV_PY.replace(
                    'env = Environment("demo")\n',
                    'env = Environment("demo")\n'
                    "from hud.capabilities import Capability\n"
                    'env.add_capability(Capability.cdp(url="ws://127.0.0.1:9222"))\n',
                ),
                encoding="utf-8",
            ),
            ValueError,
            r"non-Harbor capabilities \['cdp/1.3'\]; only ssh/mcp are convertible",
            True,
        ),
        (
            "a task the environment does not define",
            rows({"env": "demo", "id": "missing", "args": {}}),
            TypeError,
            "needs a local env defining task 'missing'",
            True,
        ),
        (
            "a slug without a usable name",
            rows({"env": "demo", "id": "solve", "args": {}, "slug": ".."}),
            ValueError,
            "does not form a usable directory name",
            True,
        ),
        (
            "two slugs naming one directory",
            rows(
                {"env": "demo", "id": "solve", "args": {"n": 1}, "slug": "suite/fix"},
                {"env": "demo", "id": "solve", "args": {"n": 2}, "slug": "suite-fix"},
            ),
            ValueError,
            "task slugs 'suite/fix' and 'suite-fix' both name the export directory 'suite-fix'",
            True,
        ),
    ],
)
async def test_export_refuses_sources_harbor_cannot_run_and_stops_what_it_started(
    case: str,
    edit: Callable[[Path], None],
    error: type[Exception],
    message: str,
    started: bool,
    tmp_path: Path,
    markers: Path,
) -> None:
    del case
    source = write_source(tmp_path / "source", {})
    edit(source)
    taskset = next(source.glob("tasks.json*"), source / "env.py")

    with pytest.raises(error, match=message):
        await export(str(taskset), tmp_path / "out")

    assert names(markers) == (["initialized", "stopped"] if started else [])


async def test_the_entrypoint_refuses_to_run_the_agent_when_task_setup_fails(
    tmp_path: Path,
) -> None:
    (task_dir,) = await export(str(write_source(tmp_path / "source", {}) / "env.py"), tmp_path)

    assert run_entrypoint(task_dir, tmp_path, start_exit=3) == snapshot(
        {
            "exit": 1,
            "stderr": "hud: task setup failed; refusing to run the agent against an unset task\n",
            "agent ran": False,
            "hud calls": [
                ["serve", "env.py:demo", "--port", "8765"],
                [
                    "task",
                    "start",
                    "solve",
                    "--args",
                    '{"n": 2, "prompt": "what\'s next?"}',
                    "--url",
                    "tcp://127.0.0.1:8765",
                ],
            ],
        }
    )


@pytest.fixture(scope="module")
def wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    wheels = tmp_path_factory.mktemp("wheels")
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(wheels)],
        cwd=REPO,
        check=True,
        capture_output=True,
    )
    return next(wheels.glob("*.whl"))


def docker(*args: str, timeout: float = 600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=timeout, check=False
    )


async def wait_for_process(container: str, command: str, deadline: float = 240) -> bool:
    """Whether ``command`` runs in ``container`` before ``deadline`` seconds pass."""
    loop = asyncio.get_running_loop()
    end = loop.time() + deadline
    while loop.time() < end:
        if command in docker("top", container).stdout:
            return True
        await asyncio.sleep(1)
    return False


@pytest.mark.e2e
@pytest.mark.docker
@pytest.mark.parametrize(
    ("answer", "exit_code", "reward"),
    [("2", 0, "1.0"), ("wrong", 0, "0.25"), ("crash", 1, None)],
)
async def test_an_exported_task_grades_its_parked_run_inside_the_built_image(
    answer: str,
    exit_code: int,
    reward: str | None,
    wheel: Path,
    tmp_path: Path,
) -> None:
    source = write_source(
        tmp_path / "source",
        {
            "Dockerfile": textwrap.dedent(f"""\
                FROM python:3.12-slim
                COPY {wheel.name} /tmp/
                RUN pip install --no-cache-dir /tmp/{wheel.name}
                WORKDIR /workspace
                COPY env.py ./
                """),
        },
    )
    shutil.copy2(wheel, source / wheel.name)
    (task_dir,) = await export(str(source / "env.py"), tmp_path / "out")
    image = f"hud-export-test:{uuid.uuid4().hex[:12]}"
    built = docker("build", "--tag", image, str(task_dir / "environment"))
    assert built.returncode == 0, built.stderr[-4000:]
    started = docker("run", "--detach", "--rm", image)
    assert started.returncode == 0, started.stderr
    container = started.stdout.strip()
    try:
        parked = await wait_for_process(container, "sleep infinity")
        assert parked, docker("logs", container).stderr[-4000:]
        assert docker("cp", str(task_dir / "tests"), f"{container}:/tests").returncode == 0
        written_answer = docker(
            "exec", container, "sh", "-c", f"printf %s {answer} > /workspace/answer.txt"
        )
        assert written_answer.returncode == 0, written_answer.stderr

        graded = docker("exec", container, "sh", "/tests/test.sh", timeout=120)
        read = docker("exec", container, "cat", "/logs/verifier/reward.txt")
    finally:
        docker("rm", "--force", container)
        docker("image", "rm", "--force", image)

    assert graded.returncode == exit_code, graded.stderr
    assert (read.stdout.strip() if read.returncode == 0 else None) == reward
