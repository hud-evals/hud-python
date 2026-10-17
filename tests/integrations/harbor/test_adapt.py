"""``harbor.adapt`` over case directories of Harbor tasks.

Each directory under ``cases/`` is a Harbor dataset; adaptation runs against the
local Docker daemon. Every case is held to what holds for any dataset (findings
by task, rows that match the written ``tasks.json``, environments copied whole,
Compose projects that resolve, no host secret written out); each case then has a
scenario for the rule it exists to exercise. Adapted tasks run end to end in
``test_controller``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from hud.integrations import harbor

if TYPE_CHECKING:
    from collections.abc import Callable

    from hud.eval import RuntimeConfig, Task
    from tests.support import HudEnv

CASES = Path(__file__).parent / "cases"
SECRET = "sk-live-secret"

# The findings each case reports, as (task, code), one entry per finding.
FINDINGS: dict[str, list[tuple[str, str]]] = {
    "findings": [
        ("artifact-absolute-destination", "harbor.invalid.task_config"),
        ("artifact-backslash-destination", "harbor.invalid.task_config"),
        ("artifact-dotdot", "harbor.invalid.task_config"),
        ("artifact-dotdot-destination", "harbor.invalid.task_config"),
        ("artifact-double-slash", "harbor.invalid.task_config"),
        ("artifact-manifest-destination", "harbor.invalid.task_config"),
        ("artifact-root", "harbor.invalid.task_config"),
        ("bad-config", "harbor.invalid.task_config"),
        ("bad-config", "harbor.invalid.task_config"),
        ("bad-toml", "harbor.invalid.task_config_toml"),
        ("compose-extends", "harbor.unsupported.compose_include_extends"),
        ("compose-include", "harbor.unsupported.compose_include_extends"),
        ("compose-not-mapping", "harbor.invalid.compose"),
        ("compose-path-escape", "harbor.invalid.compose_project_path"),
        ("healthcheck-duration", "harbor.invalid.healthcheck"),
        ("healthcheck-test", "harbor.invalid.healthcheck"),
        ("host-variable", "harbor.unsupported.host_compose_variable"),
        ("main-build-escape", "harbor.invalid.compose_project_path"),
        ("main-build-escape", "harbor.invalid.compose_main_build_escape"),
        ("main-build-escape", "harbor.invalid.missing_compose_main_dockerfile"),
        ("main-build-path-type", "harbor.invalid.compose_main_build_path"),
        ("mcp-servers", "harbor.invalid.duplicate_mcp_name"),
        ("mcp-servers", "harbor.invalid.reserved_mcp_name"),
        ("mcp-servers", "harbor.invalid.reserved_mcp_name"),
        ("mcp-servers", "harbor.invalid.mcp_url"),
        ("missing-instruction", "harbor.invalid.missing_instruction"),
        ("missing-main-dockerfile", "harbor.invalid.missing_compose_main_dockerfile"),
        ("missing-tests", "harbor.invalid.missing_tests"),
        ("missing-verifier-dockerfile", "harbor.invalid.missing_verifier_dockerfile"),
        ("multi-step", "harbor.unsupported.multi_step"),
        ("no-recipe", "harbor.invalid.environment_recipe"),
        ("non-utf8-toml", "harbor.invalid.task_config_toml"),
        ("reserved-ports", "harbor.invalid.reserved_main_port"),
        ("reserved-ports", "harbor.invalid.reserved_main_port"),
        ("reserved-ports", "harbor.invalid.reserved_main_port"),
        ("reserved-services", "harbor.invalid.reserved_compose_service"),
        ("sidecar-without-recipe", "harbor.invalid.sidecar_recipe"),
    ],
    "image-failures": [
        ("broken", "harbor.invalid.image"),
        ("missing-sidecar-image", "harbor.invalid.image"),
        ("verifier-on-arm", "harbor.invalid.image"),
    ],
}


def load_case(case: str, tmp_path: Path, hud_env: HudEnv) -> Path:
    """Copy ``case`` into ``tmp_path``, with a host secret its templates must never leak."""
    dataset = tmp_path / case
    shutil.copytree(CASES / case, dataset, symlinks=True)
    hud_env.set(HARBOR_JUDGE_KEY=SECRET)
    return dataset


def adapted(case: str, tmp_path: Path, hud_env: HudEnv) -> dict[str, Task]:
    """The rows ``case`` adapts to, by slug."""
    result = harbor.adapt(load_case(case, tmp_path, hud_env), hud_requirement="hud")
    return {row.slug or "": row for row in result.taskset}


def compose(row: Task) -> dict[str, Any]:
    """The Compose document a row runs in."""
    assert row.runtime_config is not None and row.runtime_config.compose is not None
    document = row.runtime_config.compose.document
    assert isinstance(document, Path)
    return json.loads(document.read_text("utf-8"))


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
def test_adapting_a_dataset_reports_its_findings_and_writes_runnable_projects(
    case: str, tmp_path: Path, hud_env: HudEnv
) -> None:
    dataset = load_case(case, tmp_path, hud_env)

    result = harbor.adapt(dataset, hud_requirement="hud")

    found = [
        (failure.task, finding.code) for failure in result.failures for finding in failure.findings
    ]
    assert sorted(found) == sorted(FINDINGS.get(case, []))
    projects = sorted((dataset / ".hud-adapt").iterdir())
    written = sorted(
        row["slug"]
        for project in projects
        for row in json.loads((project / "tasks.json").read_text("utf-8"))
    )
    rows = list(result.taskset)
    assert sorted(row.slug or "" for row in rows) == written
    for project in projects:
        assert_stock_compose_complete(project / "compose.yaml")
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


@pytest.mark.e2e
@pytest.mark.docker
def test_tasks_with_the_same_environment_share_one_project(tmp_path: Path, hud_env: HudEnv) -> None:
    rows = adapted("dataset-grouping", tmp_path, hud_env)

    groups: dict[str, set[str]] = {}
    for slug, row in rows.items():
        groups.setdefault(row.env, set()).add(slug)
    assert sorted(map(sorted, groups.values())) == [
        ["build-pmars", "chess-best-move"],
        ["caffe-cifar-10", "sam-cell-seg"],
        ["cancel-async-tasks"],
    ]
    pmars = rows["build-pmars"]
    assert pmars.args["task"]["artifacts"] == [
        {"service": "main", "source": "/tmp/result"},
        {
            "service": "main",
            "source": "/app/outputs",
            "destination": "results/outputs",
            "exclude": ["*.tmp", "cache"],
        },
    ]
    assert (pmars.args["task"]["verifier_timeout"], pmars.agent_config) == (
        30.0,
        {"timeout_seconds": 45.0},
    )


@pytest.mark.e2e
@pytest.mark.docker
def test_each_task_carries_its_resources_limits_and_agent_budget(
    tmp_path: Path, hud_env: HudEnv
) -> None:
    rows = adapted("resources-and-budgets", tmp_path, hud_env)

    assert {slug: placement(row.runtime_config) for slug, row in rows.items()} == {
        "gpu": {
            "resources": {
                "cpu": 4.0,
                "memory_mb": 8192,
                "storage_mb": 32768,
                "gpu": {"count": 2, "type": ["H100"]},
            }
        },
        "windows-tpu": {"resources": {"os": "windows", "tpu": {"type": "v5", "topology": "2x2"}}},
        "budget": {"limits": {"startup_timeout_s": 901}},
        "cpu-only": {},
    }
    assert rows["budget"].agent_config == {"timeout_seconds": 28800.0}


@pytest.mark.e2e
@pytest.mark.docker
def test_a_separate_verifier_runs_as_its_own_row_with_its_own_placement(
    tmp_path: Path, hud_env: HudEnv
) -> None:
    rows = adapted("separate-verifier", tmp_path, hud_env)

    verifiers = {slug: row.verifier for slug, row in rows.items()}
    assert {slug for slug, verifier in verifiers.items() if verifier} == {
        "placement",
        "plain",
        "same-placement",
    }
    actor, verifier = rows["placement"], verifiers["placement"]
    assert verifier is not None
    assert placement(actor.runtime_config) == {
        "resources": {"cpu": 2.0, "memory_mb": 2048},
        "limits": {"startup_timeout_s": 600},
    }
    assert placement(verifier.runtime_config) == {
        "resources": {"cpu": 4.0, "memory_mb": 1024, "gpu": {"count": 1, "type": ["T4"]}},
        "limits": {"startup_timeout_s": 1200},
    }
    assert actor.runtime_config is not None and actor.runtime_config.compose is not None
    assert actor.runtime_config.compose.service_access is True
    assert actor.args["task"]["collect"] == [
        {"service": "redis", "command": "redis-cli save", "timeout_sec": 10.0}
    ]
    # A verifier placed like its actor, or not placed at all, carries no placement of its own.
    for slug in ("plain", "same-placement"):
        unplaced = verifiers[slug]
        assert unplaced is not None and unplaced.runtime_config is None, slug


@pytest.mark.e2e
@pytest.mark.docker
def test_an_authored_compose_main_build_becomes_the_base_image(
    tmp_path: Path, hud_env: HudEnv
) -> None:
    rows = adapted("compose-main-build", tmp_path, hud_env)

    bases = {slug: compose(row)["services"]["hud-base"]["build"] for slug, row in rows.items()}
    assert bases == {
        "containerfile": {
            "context": "./compose-project/environment",
            "dockerfile": "Containerfile",
            "args": {"FLAVOR": "compose"},
        },
        "image-and-dockerfile": {"context": "./compose-project/environment"},
    }
    for row in rows.values():
        assert compose(row)["services"]["main"]["build"]["additional_contexts"] == {
            "hud-base": "service:hud-base"
        }


@pytest.mark.e2e
@pytest.mark.docker
def test_a_prebuilt_image_is_the_base_without_a_build(tmp_path: Path, hud_env: HudEnv) -> None:
    (row,) = adapted("prebuilt-image", tmp_path, hud_env).values()

    services = compose(row)["services"]
    assert sorted(services) == ["main"]
    args = services["main"]["build"]["args"]
    assert (args["BASE_IMAGE"], args["VERIFIER_IMAGE"]) == ("golang:1.23-alpine",) * 2


@pytest.mark.e2e
@pytest.mark.docker
def test_compose_variables_and_limits_reach_the_main_service_and_the_row(
    tmp_path: Path, hud_env: HudEnv
) -> None:
    (row,) = adapted("compose-variables", tmp_path, hud_env).values()

    main = compose(row)["services"]["main"]
    assert main["environment"] == {"FROM_DOTENV": "dotenv", "FROM_ENV": "from-task"}
    assert main["deploy"] == {"resources": {"limits": {"cpus": "2", "memory": "4096M"}}}
    assert placement(row.runtime_config) == {"resources": {"cpu": 2.0, "memory_mb": 4096}}


@pytest.mark.e2e
@pytest.mark.docker
def test_sidecars_keep_their_settings_and_their_order_around_main(
    tmp_path: Path, hud_env: HudEnv
) -> None:
    rows = adapted("sidecars", tmp_path, hud_env)

    services = compose(rows["implicit-main"])["services"]
    assert sorted(services) == ["cache", "hud-base", "main", "redis", "seed", "worker"]
    redis = services["redis"]
    assert redis["depends_on"] == {
        "main": {"condition": "service_started", "restart": True},
        "seed": {"condition": "service_completed_successfully"},
    }
    assert (redis["command"], redis["environment"], redis["healthcheck"]) == (
        ["redis-server", "--save", ""],
        {"SIDE": "car"},
        {"interval": "2s", "test": ["CMD", "redis-cli", "ping"]},
    )
    assert [services[name]["build"] for name in ("cache", "seed")] == [
        "./compose-project/environment/cache",
        "./compose-project/environment/seed",
    ]
    main = compose(rows["main-process"])["services"]["main"]
    assert (main["environment"], main["expose"]) == ({"FROM_COMPOSE": "yes"}, ["8080"])


@pytest.mark.e2e
@pytest.mark.docker
def test_a_task_that_collects_from_a_sidecar_gets_service_access(
    tmp_path: Path, hud_env: HudEnv
) -> None:
    (row,) = adapted("shared-collect-sidecar", tmp_path, hud_env).values()

    assert row.runtime_config is not None and row.runtime_config.compose is not None
    assert row.runtime_config.compose.service_access is True
    assert compose(row)["services"]["main"]["build"]["target"] == "service-access"
    assert (row.args["task"]["collect"], row.args["task"]["artifacts"]) == (
        [{"service": "redis", "command": "redis-cli save", "timeout_sec": 600.0}],
        [{"service": "redis", "source": "/data/dump.rdb"}],
    )


@pytest.mark.e2e
@pytest.mark.docker
def test_source_builds_volumes_and_env_files_are_rebased_into_the_project(
    tmp_path: Path, hud_env: HudEnv
) -> None:
    (row,) = adapted("source-builds-and-volumes", tmp_path, hud_env).values()

    document = compose(row)
    main, database = document["services"]["main"], document["services"]["database"]
    assert main["volumes"] == [
        "./compose-project/environment/main-data:/mounts/0",
        {
            "type": "bind",
            "source": "./compose-project/environment/readonly",
            "target": "/mounts/1",
            "read_only": True,
        },
        "outputs:/mounts/2",
        "./compose-project/tests:/controller/tests:ro",
    ]
    assert main["env_file"] == "./compose-project/environment/main.env"
    assert (database["build"], database["env_file"], database["volumes"]) == (
        {"context": "./compose-project/environment/database"},
        "./compose-project/environment/database/db.env",
        ["./compose-project/environment/database/data:/var/lib/postgresql/data"],
    )
    assert document["volumes"] == {"outputs": {}}


@pytest.mark.e2e
@pytest.mark.docker
def test_symlinks_in_a_task_stay_links_in_its_project(tmp_path: Path, hud_env: HudEnv) -> None:
    (row,) = adapted("symlinks", tmp_path, hud_env).values()

    assert row.runtime_config is not None and row.runtime_config.compose is not None
    root = row.runtime_config.compose.root
    assert isinstance(root, Path)
    tests = root / "compose-project" / "tests" / "task-a"
    assert os.readlink(tests / "host-file") == "/etc/hostname"
    assert os.readlink(root / "compose-project" / "environment" / "Dockerfile.link") == (
        "Dockerfile"
    )


@pytest.mark.e2e
@pytest.mark.docker
def test_an_image_docker_cannot_resolve_is_reported_on_its_task(
    tmp_path: Path, hud_env: HudEnv
) -> None:
    """A verifier that cannot build on a non-amd64 daemon says how to adapt for one platform."""
    dataset = load_case("image-failures", tmp_path, hud_env)
    host = subprocess.run(
        ["docker", "version", "--format", "{{.Server.Arch}}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    result = harbor.adapt(dataset, hud_requirement="hud")

    messages = {
        failure.task: " ".join(finding.message for finding in failure.findings)
        for failure in result.failures
    }
    assert "registry.invalid/missing" in messages["broken"]
    # Linux daemons name only the registry they could not reach; the finding names the service.
    assert {"cache", "registry.invalid"} <= set(
        re.findall(r"[\w.]+", messages["missing-sidecar-image"])
    )
    assert "--platform linux/amd64" in messages["verifier-on-arm"]
    assert ("DOCKER_DEFAULT_PLATFORM=linux/amd64" in messages["verifier-on-arm"]) is (
        host != "amd64"
    )


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
