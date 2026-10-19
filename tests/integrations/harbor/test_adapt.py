"""``harbor.adapt`` over case directories of Harbor tasks.

Each directory under ``cases/`` is a Harbor dataset; adaptation runs against the
local Docker daemon. Every case is held to what holds for any dataset (rows that
match the written ``tasks.json``, environments copied whole, Compose projects
that resolve, no host secret written out) and to its ``expected.toml``, whose
keys are all optional:

- ``findings``: ``[task, code]`` for each finding adaptation reports;
- ``mentions`` and ``mentions_off_amd64``: per task, text its finding messages
  contain (the second exactly when the Docker daemon is not amd64);
- ``projects``: the tasks that share one project, by environment;
- ``task.<slug>``: dotted paths into the row's observed fields (``placement``,
  ``agent_config``, ``task``, ``service_access``, ``services``, ``compose``,
  ``verifier.runtime_config``, ``links``), each equal to its value;
- ``base``: another case this one starts from (``assemble_case``).

Adapted tasks run end to end in ``test_controller``.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from hud.integrations import harbor
from tests.integrations.harbor.cases import assemble_case, read_expected

if TYPE_CHECKING:
    from collections.abc import Callable

    from hud.eval import RuntimeConfig, Task
    from tests.support import HudEnv

CASES = Path(__file__).parent / "cases"
SECRET = "sk-live-secret"


def load_case(case: str, tmp_path: Path, hud_env: HudEnv) -> Path:
    """Assemble ``case`` in ``tmp_path``, with a host secret its templates must never leak."""
    dataset = tmp_path / case
    assemble_case(CASES / case, dataset)
    hud_env.set(HARBOR_JUDGE_KEY=SECRET)
    return dataset


def placement(config: RuntimeConfig | None) -> dict[str, Any]:
    """A runtime config's resources and limits, as the platform reads them."""
    if config is None:
        return {}
    return config.model_dump(mode="json", exclude_none=True, include={"resources", "limits"})


def file_tree(root: Path) -> dict[str, bytes | str]:
    return {
        entry.relative_to(root).as_posix(): (
            f"-> {os.readlink(entry)}" if entry.is_symlink() else entry.read_bytes()
        )
        for entry in sorted(root.rglob("*"))
        if entry.is_symlink() or entry.is_file()
    }


def observed(row: Task) -> dict[str, Any]:
    """The fields of an adapted row that ``expected.toml`` paths point into."""
    config = row.runtime_config
    assert config is not None and config.compose is not None
    root, document = config.compose.root, config.compose.document
    assert isinstance(root, Path) and isinstance(document, Path)
    compose = json.loads(document.read_text("utf-8"))
    verifier = row.verifier
    return {
        "placement": placement(config),
        "agent_config": row.agent_config,
        "task": row.args["task"],
        "service_access": config.compose.service_access,
        "services": sorted(compose["services"]),
        "compose": compose,
        "verifier": verifier is not None
        and {
            "runtime_config": verifier.runtime_config is not None
            and placement(verifier.runtime_config)
        },
        "links": {
            entry.relative_to(root).as_posix(): os.readlink(entry)
            for entry in root.rglob("*")
            if entry.is_symlink()
        },
    }


def assert_stock_compose_complete(compose_path: Path) -> None:
    """Every service has an image, and every path a build reads resolves inside the project."""
    project_root = compose_path.parent.resolve()
    services = json.loads(compose_path.read_text("utf-8"))["services"]
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


@pytest.mark.e2e
@pytest.mark.docker
@pytest.mark.parametrize("case", sorted(path.name for path in CASES.iterdir()))
def test_adapting_a_dataset_writes_the_projects_and_findings_its_case_expects(
    case: str, tmp_path: Path, hud_env: HudEnv
) -> None:
    spec = read_expected(CASES / case)
    dataset = load_case(case, tmp_path, hud_env)

    result = harbor.adapt(dataset, hud_requirement="hud")

    found = [
        (failure.task, finding.code) for failure in result.failures for finding in failure.findings
    ]
    assert sorted(found) == sorted(map(tuple, spec.get("findings", [])))
    messages = {
        failure.task: " ".join(finding.message for finding in failure.findings)
        for failure in result.failures
    }
    for task, words in spec.get("mentions", {}).items():
        assert all(word in messages[task] for word in words), (task, messages[task])
    if off_amd64 := spec.get("mentions_off_amd64"):
        arch = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Arch}}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for task, words in off_amd64.items():
            assert all(word in messages[task] for word in words) is (arch != "amd64"), task

    projects = sorted((dataset / ".hud-adapt").iterdir())
    written = sorted(
        row["slug"]
        for project in projects
        for row in json.loads((project / "tasks.json").read_text("utf-8"))
    )
    rows = {row.slug or "": row for row in result.taskset}
    assert sorted(rows) == written
    for project in projects:
        assert_stock_compose_complete(project / "compose.yaml")
    for slug, row in rows.items():
        assert row.runtime_config is not None and row.runtime_config.compose is not None
        root = row.runtime_config.compose.root
        assert isinstance(root, Path)
        assert file_tree(root / "compose-project" / "environment") == file_tree(
            dataset / slug / "environment"
        )
    leaked = [
        path
        for path in (dataset / ".hud-adapt").rglob("*")
        if path.is_file() and not path.is_symlink() and SECRET.encode() in path.read_bytes()
    ]
    assert leaked == []

    if "projects" in spec:
        groups: dict[str, list[str]] = {}
        for slug, row in sorted(rows.items()):
            groups.setdefault(row.env, []).append(slug)
        assert sorted(groups.values()) == sorted(spec["projects"])
    for slug, fields in spec.get("task", {}).items():
        view = observed(rows[slug])
        for path, value in fields.items():
            actual: Any = view
            for key in path.split("."):
                actual = actual[key]
            assert actual == value, (slug, path)


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
    docker_state: str, message: str, tmp_path: Path, hud_env: HudEnv
) -> None:
    dataset = load_case("plain-image", tmp_path, hud_env)
    if docker_state == "daemon unreachable":
        hud_env.set(DOCKER_HOST=f"unix://{tmp_path}/no-daemon.sock")
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


@pytest.mark.e2e
@pytest.mark.docker
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
    hud_env: HudEnv,
) -> None:
    del change
    dataset = load_case("plain-image", tmp_path, hud_env)
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
