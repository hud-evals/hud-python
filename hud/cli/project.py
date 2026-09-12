"""Show and select the Project used for resource placement."""

from __future__ import annotations

from dataclasses import asdict

import typer

from hud.cli.utils.api import require_api_key
from hud.cli.utils.config import AuthScope, DirectoryLink, DirectoryState
from hud.cli.utils.output import (
    dry_run_option,
    emit_json,
    emit_quiet,
    json_option,
    output_option,
    quiet_option,
    resolve_output_mode,
    wants_json,
)
from hud.cli.utils.project import (
    Placement,
    Project,
    ProjectSource,
    list_projects,
    require_writable_placement,
    resolve_placement,
    resolve_project,
)
from hud.utils.hud_console import HUDConsole
from hud.utils.platform import PlatformClient

project_app = typer.Typer(
    name="project",
    help="Show and choose the Project for new environments and tasksets",
    add_completion=False,
    rich_markup_mode="rich",
)


@project_app.command("list")
def list_command(
    json_output: bool = json_option(),
    output: str | None = output_option(),
    quiet: bool = quiet_option(),
) -> None:
    """List all visible Projects and their canonical IDs."""
    mode = resolve_output_mode(json_output=json_output, output=output, quiet=quiet)
    require_api_key("list projects")
    projects = list_projects(PlatformClient.from_settings())
    if mode == "json":
        emit_json([asdict(project) for project in projects])
    elif mode == "quiet":
        emit_quiet([project.id for project in projects])
    else:
        console = HUDConsole()
        for project in projects:
            tags = " (default)" if project.is_default else ""
            tags += " (read-only)" if not project.can_create else ""
            console.info(f"{project.name}  {project.id}{tags}")
        if not projects:
            console.info("No projects found")


@project_app.command("create")
def create_command(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Name for the new Project"),
    description: str | None = typer.Option(None, "--description"),
    directory: str | None = typer.Option(None, "--directory", "-C"),
    no_use: bool = typer.Option(False, "--no-use", help="Create without linking this directory"),
    json_output: bool = json_option(),
    dry_run: bool = dry_run_option(),
) -> None:
    """Create a Project and link this directory unless --no-use is passed."""
    require_api_key("create a project")
    platform = PlatformClient.from_settings()
    payload = {"name": name}
    if description:
        payload["description"] = description
    if dry_run:
        if wants_json(json_output):
            emit_json({"dry_run": True, "action": "create_project", **payload})
        else:
            HUDConsole().info(f"Would create Project {name}")
        return
    state = (
        None
        if no_use
        else DirectoryState(
            AuthScope.resolve(platform), directory or ctx.meta["hud_project_directory"]
        )
    )
    if state is not None:
        state.load()
    created = Project.from_record(platform.post("/projects", json=payload))
    if state is not None:
        state.update(DirectoryLink(project_id=created.id))
    if wants_json(json_output):
        emit_json(asdict(created))
    else:
        HUDConsole().success(f"Created Project: {created.name} ({created.id})")


@project_app.command("use")
def use_command(
    ctx: typer.Context,
    ref: str = typer.Argument(..., help="Project ID from hud project list"),
    directory: str | None = typer.Option(None, "--directory", "-C"),
    json_output: bool = json_option(),
    dry_run: bool = dry_run_option(),
) -> None:
    """Link a directory to a Project in ~/.hud/config.json for this account and team."""
    require_api_key("select a project")
    platform = PlatformClient.from_settings()
    state = DirectoryState(
        AuthScope.resolve(platform), directory or ctx.meta["hud_project_directory"]
    )
    state.load()
    project = resolve_project(platform, ref)
    require_writable_placement(Placement(project, ProjectSource.FLAG))
    if not dry_run:
        state.update(DirectoryLink(project_id=project.id))
    if wants_json(json_output):
        emit_json({**asdict(project), "dry_run": dry_run})
    else:
        HUDConsole().success(
            f"{'Would use' if dry_run else 'Using'} Project: {project.name} ({project.id})"
        )


@project_app.callback(invoke_without_command=True)
def project_callback(
    ctx: typer.Context,
    directory: str = typer.Option(".", "--directory", "-C"),
    json_output: bool = json_option(),
) -> None:
    """Show the Project selected for this directory."""
    ctx.meta["hud_project_directory"] = directory
    if ctx.invoked_subcommand is not None:
        return
    require_api_key("resolve the current project")
    platform = PlatformClient.from_settings()
    platform.get("/projects", params={"limit": 1})
    state = DirectoryState(AuthScope.resolve(platform), directory)
    placement = resolve_placement(platform, state.load(), flag=None)
    if wants_json(json_output):
        emit_json(
            {
                "project": asdict(placement.project) if placement.project else None,
                "source": placement.source.value,
            }
        )
    else:
        HUDConsole().info(f"Project: {placement.label}")
