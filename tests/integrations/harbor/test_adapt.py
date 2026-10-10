"""``harbor.adapt`` over case directories of Harbor tasks, against the fake docker.

Each directory under ``cases/`` is a Harbor dataset plus a ``docker.json`` that
says what the docker daemon holds: the OCI config each build context and pulled
image produces, the server platform, and the commands that fail. The scenario
adapts a copy of the dataset and compares what a user and the runtime observe
(the failures, the generated tree and each project's ``tasks.json``) with the
case's ``expected.json``. A case about Compose output or docker commands lists
``"compose"`` or ``"docker"`` under ``observe`` in its ``docker.json`` to record
those too.
Record a new case with ``--inline-snapshot=create``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from inline_snapshot import external_file

from hud.integrations import harbor

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.harness import FakeDocker, HudEnv

CASES = Path(__file__).parent / "cases"
SECRET = "sk-live-secret"
HASH = re.compile(r"\b[0-9a-f]{12}(?:[0-9a-f]{4})?\b")
RESOLVE_DIR = re.compile(r"[^\s\"']*/hud-harbor-resolve-[^/\s\"']+")


def load_case(
    case: str, tmp_path: Path, fake_docker: FakeDocker, hud_env: HudEnv
) -> tuple[Path, set[str]]:
    """Copy ``case`` into ``tmp_path``, load its ``docker.json``, and return what it observes."""
    dataset = tmp_path / case
    shutil.copytree(
        CASES / case,
        dataset,
        symlinks=True,
        ignore=shutil.ignore_patterns("docker.json", "expected.json"),
    )
    spec = json.loads((CASES / case / "docker.json").read_text("utf-8"))
    fake_docker.images(
        dataset,
        builds=spec.get("builds"),
        pulls=spec.get("pulls"),
        platform=spec.get("platform", "linux/amd64"),
    )
    for failure in spec.get("fail", []):
        fake_docker.on(failure["match"], stderr=failure["stderr"], exit=1)
    # A host value a template names must never reach anything adapt writes.
    hud_env.set(HARBOR_JUDGE_KEY=SECRET)
    return dataset, set(spec.get("observe", []))


def mask(document: Any) -> Any:
    """Replace each distinct content hash with ``<hN>``, numbered by first appearance."""
    text = json.dumps(document, sort_keys=True)
    labels: dict[str, str] = {}
    for token in HASH.findall(text):
        labels.setdefault(token, f"<h{len(labels) + 1}>")
    return json.loads(HASH.sub(lambda match: labels[match.group()], text))


def listing(root: Path) -> list[str]:
    entries = []
    for entry in sorted(root.rglob("*")):
        relative = entry.relative_to(root).as_posix()
        if entry.is_symlink():
            entries.append(f"{relative} -> {os.readlink(entry)}")
        elif entry.is_file():
            entries.append(relative)
    return entries


def file_tree(root: Path) -> dict[str, bytes | str]:
    return {
        entry.relative_to(root).as_posix(): (
            f"-> {os.readlink(entry)}" if entry.is_symlink() else entry.read_bytes()
        )
        for entry in sorted(root.rglob("*"))
        if entry.is_symlink() or entry.is_file()
    }


def assert_stock_compose_complete(compose_path: Path) -> dict[str, Any]:
    """Every service has an image, and every path a build reads resolves inside the project."""
    project_root = compose_path.parent.resolve()
    project = json.loads(compose_path.read_text("utf-8"))
    services = project["services"]
    for name, service in services.items():
        assert service.get("image"), name
        for volume in service.get("volumes", []):
            if isinstance(volume, str) and volume.endswith(":/controller/tests:ro"):
                tests = (project_root / volume.partition(":")[0]).resolve()
                assert tests.is_relative_to(project_root) and tests.is_dir(), (name, tests)
        build = service.get("build")
        if build is None:
            continue
        build = {"context": build} if isinstance(build, str) else build
        context = (project_root / build.get("context", ".")).resolve()
        assert context.is_relative_to(project_root) and context.is_dir(), (name, context)
        assert (context / build.get("dockerfile", "Dockerfile")).is_file(), name
        for target in build.get("additional_contexts", {}).values():
            if target.startswith("service:"):
                assert target.removeprefix("service:") in services, (name, target)
            else:
                named = (project_root / target).resolve()
                assert named.is_relative_to(project_root) and named.is_dir(), (name, named)
    return project


def observe(
    result: harbor.AdaptResult, dataset: Path, fake_docker: FakeDocker, extra: set[str]
) -> Any:
    def scrub(text: str) -> str:
        return RESOLVE_DIR.sub("<resolve>", text.replace(str(dataset), "<case>"))

    projects: dict[str, Any] = {}
    adapted = dataset / ".hud-adapt"
    for context in sorted(adapted.iterdir()) if adapted.is_dir() else []:
        compose = assert_stock_compose_complete(context / "compose.yaml")
        projects[context.name] = {
            "tree": listing(context),
            **({"compose.yaml": compose} if "compose" in extra else {}),
            "tasks.json": json.loads((context / "tasks.json").read_text("utf-8")),
        }
    docker = [
        {
            "argv": scrub(call.command),
            **(
                {
                    "files": {
                        Path(path).name: json.loads(content) for path, content in call.files.items()
                    }
                }
                if call.files
                else {}
            ),
        }
        for call in fake_docker.calls
    ]
    return mask(
        {
            "failures": [
                [failure.task, finding.code, finding.kind, scrub(finding.message)]
                for failure in result.failures
                for finding in failure.findings
            ],
            "projects": projects,
            **({"docker": docker} if "docker" in extra else {}),
        }
    )


@pytest.mark.parametrize("case", sorted(path.name for path in CASES.iterdir()))
def test_adapting_a_harbor_dataset_writes_its_projects_and_reports_its_failures(
    case: str, tmp_path: Path, fake_docker: FakeDocker, hud_env: HudEnv
) -> None:
    dataset, extra = load_case(case, tmp_path, fake_docker, hud_env)

    result = harbor.adapt(dataset, hud_requirement="hud")

    assert observe(result, dataset, fake_docker, extra) == external_file(
        CASES / case / "expected.json", format=".json"
    )
    rows = list(result.taskset)
    assert sorted(row.slug or "" for row in rows) == sorted(
        row["slug"]
        for context in (dataset / ".hud-adapt").glob("*")
        for row in json.loads((context / "tasks.json").read_text("utf-8"))
    )
    for row in rows:
        assert row.runtime_config is not None and row.runtime_config.compose is not None
        root = row.runtime_config.compose.root
        assert isinstance(root, Path)
        assert file_tree(root / "compose-project" / "environment") == file_tree(
            dataset / (row.slug or "") / "environment"
        )
    leaked = [
        path
        for path in (dataset / ".hud-adapt").rglob("*")
        if path.is_file() and not path.is_symlink() and SECRET.encode() in path.read_bytes()
    ]
    assert leaked == []


def test_adapt_raises_when_the_source_holds_no_harbor_tasks(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()

    for source in (tmp_path / "empty", tmp_path / "nowhere"):
        with pytest.raises(ValueError, match="no Harbor tasks found"):
            harbor.adapt(source)


@pytest.mark.parametrize(
    ("docker_state", "message"),
    [
        ("daemon unreachable", "requires a reachable Docker daemon: docker version"),
        ("no docker executable", "requires Docker to resolve authored images"),
    ],
)
def test_adapt_aborts_without_a_docker_daemon_instead_of_failing_tasks(
    docker_state: str,
    message: str,
    tmp_path: Path,
    fake_docker: FakeDocker,
    hud_env: HudEnv,
) -> None:
    dataset, _ = load_case("plain-image", tmp_path, fake_docker, hud_env)
    if docker_state == "daemon unreachable":
        fake_docker.on(r"^version", stderr="Cannot connect to the Docker daemon", exit=1)
    else:
        (tmp_path / "empty-path").mkdir()
        hud_env.set(PATH=str(tmp_path / "empty-path"))

    with pytest.raises(RuntimeError, match=message):
        harbor.adapt(dataset)


def edit(relative: str, content: str) -> Callable[[Path], str]:
    def apply(task: Path) -> str:
        (task / relative).write_text(content, encoding="utf-8")
        return "hud"

    return apply


def wheel(task: Path) -> str:
    built = task.parent.parent / "hud-0.0.0-py3-none-any.whl"
    built.write_bytes(b"wheel")
    return str(built)


@pytest.mark.parametrize(
    ("change", "apply", "rebuilds"),
    [
        ("an edited instruction", edit("instruction.md", "Second instruction\n"), False),
        ("edited inline tests", edit("tests/test.sh", "#!/bin/sh\nexit 1\n"), False),
        ("a new verifier timeout", edit("task.toml", "[verifier]\ntimeout_sec = 5\n"), False),
        ("new content behind an environment symlink", edit("../outside.txt", "changed\n"), False),
        ("an edited environment file", edit("environment/ignored.txt", "changed\n"), True),
        ("another hud release", lambda task: "hud==0.0.1", True),
        ("a hud wheel", wheel, True),
    ],
)
def test_the_environment_image_identity_follows_only_what_the_image_contains(
    change: str,
    apply: Callable[[Path], str],
    rebuilds: bool,
    tmp_path: Path,
    fake_docker: FakeDocker,
    hud_env: HudEnv,
) -> None:
    del change
    dataset, _ = load_case("plain-image", tmp_path, fake_docker, hud_env)
    task = dataset / "task-a"
    (dataset / "outside.txt").write_text("first\n", encoding="utf-8")
    (task / "environment" / "outside").symlink_to(dataset / "outside.txt")

    def images(requirement: str) -> tuple[dict[str, Any], Path]:
        (row,) = harbor.adapt(dataset, hud_requirement=requirement).taskset
        assert row.runtime_config is not None and row.runtime_config.compose is not None
        assert isinstance(row.runtime_config.compose.document, Path)
        compose = json.loads(row.runtime_config.compose.document.read_text("utf-8"))
        return compose["services"]["main"], row.runtime_config.compose.document.parent

    before, _ = images("hud")
    after, project = images(apply(task))

    assert (after["image"] != before["image"]) is rebuilds
    requirement = after["build"]["args"]["HUD_REQUIREMENT"]
    packages = sorted(path.name for path in (project / "compose-project/main/packages").iterdir())
    if requirement.endswith(".whl"):
        assert (requirement, packages) == (
            "/controller/packages/hud-0.0.0-py3-none-any.whl",
            ["hud-0.0.0-py3-none-any.whl"],
        )
    else:
        assert packages == []
