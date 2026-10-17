"""``harbor.export`` writes Harbor task folders whose scripts drive the ``hud`` CLI.

The default lane checks what export writes for each source form: one folder per
row, the task Harbor reads, and the source image with HUD's entrypoint as its
final startup directives. The docker lane builds the exported image, starts it
the way Harbor does, and runs ``tests/test.sh`` inside it.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import textwrap
import tomllib
import uuid
from typing import TYPE_CHECKING, Any

import pytest

from hud.integrations.harbor import export

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tests.support import HudEnv

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
ENTRYPOINT = ['ENTRYPOINT ["/hud_entrypoint.sh"]', 'CMD ["sh", "-c", "sleep infinity"]']
PROMPT_ARGS = {"n": 2, "prompt": "what's next?"}


def write_source(root: Path, files: dict[str, str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for relative, content in {"env.py": ENV_PY, "Dockerfile": DOCKERFILE, **files}.items():
        (root / relative).write_text(content, encoding="utf-8")
    return root


def rows(*entries: dict[str, Any]) -> Callable[[Path], None]:
    """Write ``entries`` as the source's taskset: JSON for one, JSONL for several."""

    def write(source: Path) -> None:
        if len(entries) == 1:
            (source / "tasks.json").write_text(json.dumps(list(entries)), encoding="utf-8")
        else:
            lines = "\n".join(json.dumps(entry) for entry in entries)
            (source / "tasks.jsonl").write_text(lines, encoding="utf-8")

    return write


def main_thread_guard(source: Path) -> None:
    env_py = source / "env.py"
    env_py.write_text(
        "import threading\nassert threading.current_thread() is threading.main_thread()\n"
        + env_py.read_text("utf-8"),
        encoding="utf-8",
    )


def bound_as_bench(source: Path) -> None:
    (source / "tasks.py").write_text(
        ENV_PY.replace("env = Environment", "bench = Environment").replace("@env.", "@bench."),
        encoding="utf-8",
    )


def names(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir()) if directory.exists() else []


@pytest.fixture
def markers(tmp_path: Path, hud_env: HudEnv) -> Path:
    """Where the source environment's initialize and shutdown hooks leave a file."""
    directory = tmp_path / "markers"
    directory.mkdir()
    hud_env.set(EXPORT_TEST_MARKERS=str(directory))
    return directory


SOURCES: dict[str, tuple[str, Callable[[Path], None] | None, dict[str, dict[str, Any]]]] = {
    "a python source, imported on the main thread": (
        "env.py",
        main_thread_guard,
        {"solve-4080368a": PROMPT_ARGS},
    ),
    "an environment bound under another name": (
        "tasks.py",
        bound_as_bench,
        {"solve-4080368a": PROMPT_ARGS},
    ),
    "a json taskset with a nested slug": (
        "tasks.json",
        rows({"env": "demo", "id": "solve", "args": {"n": 3}, "slug": "suite/fix"}),
        {"suite-fix": {"n": 3}},
    ),
    "a jsonl taskset": (
        "tasks.jsonl",
        rows(
            {"env": "demo", "id": "solve", "args": {"n": 4}, "slug": "four"},
            {"env": "demo", "id": "solve", "args": {"n": 5}, "slug": "five"},
        ),
        {"four": {"n": 4}, "five": {"n": 5}},
    ),
}


@pytest.mark.parametrize(("source_file", "edit", "folders"), SOURCES.values(), ids=SOURCES.keys())
async def test_export_writes_one_harbor_task_per_row(
    source_file: str,
    edit: Callable[[Path], None] | None,
    folders: dict[str, dict[str, Any]],
    tmp_path: Path,
    markers: Path,
) -> None:
    source = write_source(tmp_path / "source", {})
    if edit is not None:
        edit(source)

    created = await export(str(source / source_file), tmp_path / "out")

    exported = {}
    for task_dir in created:
        task = tomllib.loads((task_dir / "task.toml").read_text("utf-8"))
        exported[task_dir.name] = json.loads(task["metadata"]["hud_args"])
        assert task["metadata"]["hud_task"] == "solve"
        assert (task_dir / "tests" / "test.sh").is_file()
        assert (task_dir / "environment" / "hud_entrypoint.sh").is_file()
    assert exported == folders
    # Export loads the source to read its rows, and stops what it started.
    assert names(markers) == ["initialized", "stopped"]


def last_directives(dockerfile: Path) -> list[str]:
    return [line for line in dockerfile.read_text("utf-8").splitlines() if line][-2:]


@pytest.mark.parametrize(
    "dockerfile",
    [
        pytest.param(DOCKERFILE, id="a-plain-image"),
        pytest.param(
            "FROM python:3.12-slim\nRUN useradd -m app\nUSER app\n"
            'CMD ["python", \\\n    "-m", \\\n    "app"]\n',
            id="a-non-root-image-with-a-multiline-cmd",
        ),
    ],
)
async def test_the_exported_image_is_the_source_image_started_by_hud(
    dockerfile: str, tmp_path: Path, markers: Path
) -> None:
    source = write_source(tmp_path / "source", {"Dockerfile": dockerfile})

    (task_dir,) = await export(str(source / "env.py"), tmp_path / "out")

    exported = (task_dir / "environment" / "Dockerfile").read_text("utf-8")
    assert exported.startswith(dockerfile)
    assert last_directives(task_dir / "environment" / "Dockerfile") == ENTRYPOINT


async def test_a_hud_dockerfile_is_exported_in_place_of_the_plain_one(
    tmp_path: Path, markers: Path
) -> None:
    hud_dockerfile = "FROM python:3.12-slim AS hud\nRUN pip install hud\n"
    source = write_source(tmp_path / "source", {"Dockerfile.hud": hud_dockerfile})

    (task_dir,) = await export(str(source / "env.py"), tmp_path / "out")

    assert (task_dir / "environment" / "Dockerfile").read_text("utf-8").startswith(hud_dockerfile)


async def test_a_restrictive_dockerignore_still_admits_the_entrypoint(
    tmp_path: Path, markers: Path
) -> None:
    source = write_source(tmp_path / "source", {".dockerignore": "*\n", "package.json": "{}"})

    (task_dir,) = await export(str(source / "env.py"), tmp_path / "out")

    ignored = (task_dir / "environment" / ".dockerignore").read_text("utf-8").splitlines()
    assert (ignored[0], ignored[-1]) == ("*", "!hud_entrypoint.sh")
    assert (task_dir / "environment" / "package.json").is_file()


async def test_the_instruction_is_the_prompt_and_where_to_answer(
    tmp_path: Path, markers: Path
) -> None:
    source = write_source(tmp_path / "source", {})

    (task_dir,) = await export(str(source / "env.py"), tmp_path / "out", answer_file="/app/out.txt")

    instruction = (task_dir / "instruction.md").read_text("utf-8")
    assert instruction.startswith("solve 2: what's next?")
    assert "`/app/out.txt`" in instruction


def with_cdp(source: Path) -> None:
    (source / "env.py").write_text(
        ENV_PY.replace(
            'env = Environment("demo")\n',
            'env = Environment("demo")\n'
            "from hud.capabilities import Capability\n"
            'env.add_capability(Capability.cdp(url="ws://127.0.0.1:9222"))\n',
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("edit", "error", "started"),
    [
        pytest.param(
            lambda source: (source / "Dockerfile").unlink(),
            FileNotFoundError,
            False,
            id="no-dockerfile",
        ),
        pytest.param(with_cdp, ValueError, True, id="a-capability-harbor-cannot-carry"),
        pytest.param(
            rows({"env": "demo", "id": "missing", "args": {}}),
            TypeError,
            True,
            id="a-task-the-environment-does-not-define",
        ),
        pytest.param(
            rows({"env": "demo", "id": "solve", "args": {}, "slug": ".."}),
            ValueError,
            True,
            id="a-slug-without-a-usable-name",
        ),
        pytest.param(
            rows(
                {"env": "demo", "id": "solve", "args": {"n": 1}, "slug": "suite/fix"},
                {"env": "demo", "id": "solve", "args": {"n": 2}, "slug": "suite-fix"},
            ),
            ValueError,
            True,
            id="two-slugs-naming-one-directory",
        ),
    ],
)
async def test_export_refuses_sources_harbor_cannot_run_and_stops_what_it_started(
    edit: Callable[[Path], None],
    error: type[Exception],
    started: bool,
    tmp_path: Path,
    markers: Path,
) -> None:
    source = write_source(tmp_path / "source", {})
    edit(source)
    taskset = next(source.glob("tasks.json*"), source / "env.py")

    with pytest.raises(error):
        await export(str(taskset), tmp_path / "out")

    assert names(tmp_path / "out") == []
    assert names(markers) == (["initialized", "stopped"] if started else [])


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


async def built(
    source_file: str, edit: Callable[[Path], None] | None, sdk_wheel: Path, tmp_path: Path
) -> tuple[Path, str]:
    """Export a source whose image installs this checkout, and build its first task's image."""
    source = write_source(
        tmp_path / "source",
        {
            "Dockerfile": textwrap.dedent(f"""\
                FROM python:3.12-slim
                COPY {sdk_wheel.name} /tmp/
                RUN pip install --no-cache-dir /tmp/{sdk_wheel.name}
                WORKDIR /workspace
                COPY . ./
                """),
        },
    )
    if edit is not None:
        edit(source)
    shutil.copy2(sdk_wheel, source / sdk_wheel.name)
    task_dir, *_ = await export(str(source / source_file), tmp_path / "out")
    image = f"hud-export-test:{uuid.uuid4().hex[:12]}"
    build = docker("build", "--tag", image, str(task_dir / "environment"))
    assert build.returncode == 0, build.stderr[-4000:]
    return task_dir, image


async def grade(task_dir: Path, image: str, answer: str) -> tuple[int, str | None]:
    """Start ``image`` as Harbor does, answer, run ``tests/test.sh``; its exit and reward."""
    started = docker("run", "--detach", "--rm", image)
    assert started.returncode == 0, started.stderr
    container = started.stdout.strip()
    try:
        parked = await wait_for_process(container, "sleep infinity")
        assert parked, docker("logs", container).stderr[-4000:]
        assert docker("cp", str(task_dir / "tests"), f"{container}:/tests").returncode == 0
        written = docker(
            "exec", container, "sh", "-c", f"printf %s {answer} > /workspace/answer.txt"
        )
        assert written.returncode == 0, written.stderr
        graded = docker("exec", container, "sh", "/tests/test.sh", timeout=120)
        read = docker("exec", container, "cat", "/logs/verifier/reward.txt")
    finally:
        docker("rm", "--force", container)
    return graded.returncode, read.stdout.strip() if read.returncode == 0 else None


@pytest.mark.e2e
@pytest.mark.docker
@pytest.mark.parametrize(
    ("answer", "exit_code", "reward"),
    [("2", 0, "1.0"), ("wrong", 0, "0.25"), ("crash", 1, None)],
)
async def test_an_exported_task_grades_its_parked_run_inside_the_built_image(
    answer: str, exit_code: int, reward: str | None, sdk_wheel: Path, tmp_path: Path
) -> None:
    task_dir, image = await built("env.py", None, sdk_wheel, tmp_path)
    try:
        graded = await grade(task_dir, image, answer)
    finally:
        docker("image", "rm", "--force", image)

    assert graded == (exit_code, reward)


@pytest.mark.e2e
@pytest.mark.docker
@pytest.mark.parametrize(
    ("source_file", "edit", "answer"),
    [
        pytest.param("tasks.py", bound_as_bench, "2", id="an-environment-bound-under-another-name"),
        pytest.param(
            "tasks.jsonl",
            rows(
                {"env": "demo", "id": "solve", "args": {"n": 4}, "slug": "four"},
                {"env": "demo", "id": "solve", "args": {"n": 5}, "slug": "five"},
            ),
            "4",
            id="a-taskset-file",
        ),
    ],
)
async def test_each_source_form_serves_its_task_inside_the_built_image(
    source_file: str, edit: Callable[[Path], None], answer: str, sdk_wheel: Path, tmp_path: Path
) -> None:
    task_dir, image = await built(source_file, edit, sdk_wheel, tmp_path)
    try:
        graded = await grade(task_dir, image, answer)
    finally:
        docker("image", "rm", "--force", image)

    assert graded == (0, "1.0")


@pytest.mark.e2e
@pytest.mark.docker
async def test_an_image_whose_task_setup_fails_never_starts_the_agent(
    sdk_wheel: Path, tmp_path: Path
) -> None:
    def setup_breaks_in_the_image(source: Path) -> None:
        # Export starts the task to read its prompt, so only the image may break it.
        env_py = source / "env.py"
        env_py.write_text(
            env_py.read_text("utf-8").replace(
                '    answer = yield f"solve {n}: {prompt}"',
                '    if os.environ.get("BREAK_SETUP"):\n'
                '        raise RuntimeError("setup broke")\n'
                '    answer = yield f"solve {n}: {prompt}"',
            ),
            encoding="utf-8",
        )
        with (source / "Dockerfile").open("a", encoding="utf-8") as dockerfile:
            dockerfile.write("ENV BREAK_SETUP=1\n")

    _, image = await built("env.py", setup_breaks_in_the_image, sdk_wheel, tmp_path)
    try:
        refused = docker("run", "--rm", image, "true", timeout=240)
        # The same image, its setup working, does run the agent's command.
        ran = docker("run", "--rm", "--env", "BREAK_SETUP=", image, "true", timeout=240)
    finally:
        docker("image", "rm", "--force", image)

    assert (refused.returncode != 0, ran.returncode) == (True, 0), ran.stderr[-2000:]
