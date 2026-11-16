"""Load Harbor tasks to run natively, each in its authored image or Compose project."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hud.eval.taskset import Taskset
from hud.runtime.compose import ComposeConfig, ComposeProject
from hud.runtime.core import (
    RuntimeBuild,
    RuntimeConfig,
    RuntimeHealthcheck,
    RuntimeProcess,
)
from hud.utils.naming import normalize_environment_name

from .adapt import (
    COMPOSE_FILENAME,
    AdaptFailure,
    AdaptFinding,
    AdaptResult,
    inspect_task,
    task_dirs,
)
from .grading import env_refs
from .session import mcp_port
from .task import HarborTask, HarborVerifierTask

if TYPE_CHECKING:
    from pathlib import Path

    from hud.runtime.core import RuntimePort

    from .adapt import HarborSource
    from .config import HealthcheckConfig

# Harbor keeps a task's box up for its agent: the image's entrypoint runs with this command.
KEEP_ALIVE = ["sh", "-c", "sleep infinity"]


def _unsupported(source: HarborSource) -> list[AdaptFinding]:
    """What ``source`` asks for that running natively does not do yet."""
    config = source.config
    network_policy = (
        config.environment.network_mode != "public"
        or config.agent.network_mode not in (None, "public")
        or config.verifier.network_mode not in (None, "public")
    )
    if not network_policy:
        return []
    return [
        AdaptFinding(
            code="harbor.unsupported.native_network_policy",
            kind="contract",
            message="network policies run only adapted for now",
        )
    ]


def _healthcheck(healthcheck: HealthcheckConfig | None) -> RuntimeHealthcheck | None:
    if healthcheck is None:
        return None
    return RuntimeHealthcheck(
        command=healthcheck.command,
        interval_s=healthcheck.interval_sec,
        timeout_s=healthcheck.timeout_sec,
        start_period_s=healthcheck.start_period_sec,
        start_interval_s=healthcheck.start_interval_sec,
        retries=healthcheck.retries,
    )


def _recipe(source: HarborSource) -> ComposeConfig:
    """``source``'s Compose project as Harbor runs it: unless ``main`` says otherwise, built
    from the environment's Dockerfile or run from the prebuilt image."""
    recipe = source.compose.model_copy(deep=True)
    main = recipe.services["main"]
    if main.build is None and source.dockerfile.is_file():
        main = main.model_copy(update={"build": {"context": "."}, "image": source.base_image})
    elif main.build is None and main.image is None:
        main = main.model_copy(update={"image": source.base_image})
    recipe.services["main"] = main
    return recipe


def _placement(source: HarborSource) -> tuple[RuntimeConfig, dict[str, RuntimePort]]:
    """``source``'s box as Harbor runs it, and the ports its MCP servers are published from."""
    environment = source.config.environment
    environment_dir = source.path / "environment"
    recipe = _recipe(source) if (environment_dir / COMPOSE_FILENAME).is_file() else None
    services = set(recipe.services) if recipe is not None else {"main"}
    mcp_ports = {
        server.name: port
        for server in environment.mcp_servers
        if (port := mcp_port(server, services)) is not None
    }
    config = RuntimeConfig(
        image=(
            None
            if recipe is not None
            else RuntimeBuild(context=source.dockerfile.parent)
            if source.dockerfile.is_file()
            else source.base_image
        ),
        compose=None if recipe is None else ComposeProject(document=recipe, root=environment_dir),
        process=RuntimeProcess(
            command=(recipe.services["main"].command if recipe is not None else None) or KEEP_ALIVE,
            env=env_refs(environment.env),
            healthcheck=_healthcheck(environment.healthcheck),
        ),
        ports=list(mcp_ports.values()) or None,
        resources=source.resources,
        limits=environment.runtime_limits,
    )
    return config, mcp_ports


def _verifier(source: HarborSource, env: str) -> HarborVerifierTask | None:
    """``source``'s separate verifier, in a box built from ``tests/``, if it has one."""
    config = source.config
    if not config.verifier.separate:
        return None
    environment = config.verifier.environment
    return HarborVerifierTask(
        env=env,
        id="verify",
        path=source.path,
        config=config,
        runtime_config=RuntimeConfig(
            image=RuntimeBuild(context=source.path / "tests"),
            process=RuntimeProcess(command=KEEP_ALIVE),
            resources=environment.runtime_resources
            if environment is not None
            else source.resources,
            limits=(environment or config.environment).runtime_limits,
        ),
    )


def load(path: str | Path) -> AdaptResult:
    """Load Harbor tasks to run natively: each in its authored image or Compose project, run
    by Harbor's lifecycle from this process.

    Loading only reads the task directories; the runtime builds each image when it places
    the task. Tasks needing what native runs do not do yet come back as failures.
    """
    directories, dataset = task_dirs(path)
    base_name = normalize_environment_name(dataset.name, default="harbor")
    rows: list[HarborTask] = []
    failures: list[AdaptFailure] = []
    for task_dir in directories:
        source, findings = inspect_task(task_dir)
        if source is None or (findings := tuple(_unsupported(source))):
            failures.append(AdaptFailure(task=task_dir.name, path=task_dir, findings=findings))
            continue
        config = source.config
        columns = dict(config.metadata)
        if config.task.keywords:
            columns.setdefault("keywords", config.task.keywords)
        env = f"{base_name}-{source.environment_hash[:12]}"
        runtime_config, mcp_ports = _placement(source)
        rows.append(
            HarborTask(
                env=env,
                id=task_dir.name,
                path=task_dir,
                config=config,
                mcp_ports=mcp_ports,
                agent_config=(
                    {"timeout_seconds": config.agent.timeout_sec}
                    if config.agent.timeout_sec is not None
                    else None
                ),
                columns=columns or None,
                runtime_config=runtime_config,
                verifier=_verifier(source, env),
            )
        )
    return AdaptResult(taskset=Taskset(dataset.name, rows), failures=tuple(failures))
