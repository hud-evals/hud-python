"""``harbor.export`` writes Harbor task folders whose scripts drive the ``hud`` CLI.

Each directory under ``sources/`` is a HUD source; ``expected.toml`` says what
exporting it yields. Keys, all optional:

- ``export``: the file export reads (default ``env.py``);
- ``answer_file``: where the instruction tells the agent to answer;
- ``folders.<name>``: a folder export writes, with the ``task`` and ``args`` its
  ``task.toml`` carries and the ``prompt`` its instruction starts with;
- ``folders.<name>.grade``: in the docker lane, an ``answer`` written into the
  started image and the ``exit`` and ``reward`` (absent: none) of ``tests/test.sh``;
- ``folders.<name>.run``: in the docker lane, whether the agent's command
  ``succeeds`` in the image under ``env``;
- ``refusal``: the error ``type`` and ``message`` export raises instead, and
  whether it ``started`` the source environment first;
- ``base``: another source this one starts from (``assemble_case``).

Every exported folder is held to what holds for any source: the source image
with HUD's entrypoint as its final startup directives, the source's files in its
build context, and an instruction naming where to answer. Export stops every
environment it starts.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
import tomllib
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from hud.integrations.harbor import export
from tests.support import assemble_case, read_expected

if TYPE_CHECKING:
    from tests.support import HudEnv

SOURCES = Path(__file__).parent / "sources"
ENTRYPOINT = ['ENTRYPOINT ["/hud_entrypoint.sh"]', 'CMD ["sh", "-c", "sleep infinity"]']
CASES = sorted(path.parent.name for path in SOURCES.glob("*/expected.toml"))
DOCKER_CASES = [
    case
    for case in CASES
    if any(
        "grade" in folder or "run" in folder
        for folder in read_expected(SOURCES / case).get("folders", {}).values()
    )
]


def names(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir()) if directory.exists() else []


@pytest.fixture
def markers(tmp_path: Path, hud_env: HudEnv) -> Path:
    """Where the source environment's initialize and shutdown hooks leave a file."""
    directory = tmp_path / "markers"
    directory.mkdir()
    hud_env.set(EXPORT_TEST_MARKERS=str(directory))
    return directory


@pytest.mark.parametrize("case", CASES)
async def test_exporting_a_source_writes_the_folders_its_case_expects(
    case: str, tmp_path: Path, markers: Path
) -> None:
    spec = read_expected(SOURCES / case)
    source = tmp_path / "source"
    assemble_case(SOURCES / case, source)
    exported = source / spec.get("export", "env.py")
    answer_file = spec.get("answer_file", "/workspace/answer.txt")

    if refusal := spec.get("refusal"):
        with pytest.raises(Exception, match=re.escape(refusal["message"])) as raised:
            await export(str(exported), tmp_path / "out", answer_file=answer_file)
        assert type(raised.value).__name__ == refusal["type"]
        assert names(tmp_path / "out") == []
        assert names(markers) == (["initialized", "stopped"] if refusal["started"] else [])
        return

    created = await export(str(exported), tmp_path / "out", answer_file=answer_file)

    folders = spec["folders"]
    assert sorted(task_dir.name for task_dir in created) == sorted(folders)
    image = next(
        source / name for name in ("Dockerfile.hud", "Dockerfile") if (source / name).is_file()
    ).read_text("utf-8")
    rewritten = {"Dockerfile", "Dockerfile.hud", ".dockerignore"}
    if exported.suffix != ".py":
        rewritten.add(exported.name)
    copied = [path for path in source.iterdir() if path.is_file() and path.name not in rewritten]
    for task_dir in created:
        expected = folders[task_dir.name]
        task = tomllib.loads((task_dir / "task.toml").read_text("utf-8"))
        assert task["metadata"]["hud_task"] == expected["task"]
        assert json.loads(task["metadata"]["hud_args"]) == expected["args"]
        assert (task_dir / "tests" / "test.sh").is_file()
        environment = task_dir / "environment"
        assert (environment / "hud_entrypoint.sh").is_file()
        dockerfile = (environment / "Dockerfile").read_text("utf-8")
        assert dockerfile.startswith(image)
        assert [line for line in dockerfile.splitlines() if line][-2:] == ENTRYPOINT
        if (source / ".dockerignore").is_file():
            ignored = (environment / ".dockerignore").read_text("utf-8")
            assert ignored.startswith((source / ".dockerignore").read_text("utf-8"))
            assert ignored.splitlines()[-1] == "!hud_entrypoint.sh"
        for path in copied:
            assert (environment / path.name).read_bytes() == path.read_bytes(), path.name
        instruction = (task_dir / "instruction.md").read_text("utf-8")
        assert instruction.startswith(expected["prompt"])
        assert f"`{answer_file}`" in instruction
    # Export loads the source to read its rows, and stops what it started.
    assert names(markers) == ["initialized", "stopped"]


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
@pytest.mark.parametrize("case", DOCKER_CASES)
async def test_an_exported_task_runs_and_grades_inside_its_built_image(
    case: str, sdk_wheel: Path, tmp_path: Path
) -> None:
    spec = read_expected(SOURCES / case)
    source = tmp_path / "source"
    assemble_case(SOURCES / case, source)
    shutil.copy2(sdk_wheel, source / sdk_wheel.name)
    created = await export(str(source / spec.get("export", "env.py")), tmp_path / "out")

    for task_dir in created:
        expected: dict[str, Any] = spec["folders"][task_dir.name]
        if not ("grade" in expected or "run" in expected):
            continue
        image = f"hud-export-test:{uuid.uuid4().hex[:12]}"
        build = docker("build", "--tag", image, str(task_dir / "environment"))
        assert build.returncode == 0, build.stderr[-4000:]
        try:
            for row in expected.get("grade", []):
                graded = await grade(task_dir, image, row["answer"])
                assert graded == (row["exit"], row.get("reward")), row
            for row in expected.get("run", []):
                env = [f"--env={name}={value}" for name, value in row["env"].items()]
                ran = docker("run", "--rm", *env, image, "true", timeout=240)
                assert (ran.returncode == 0) is row["succeeds"], ran.stderr[-2000:]
        finally:
            docker("image", "rm", "--force", image)
