"""``hud project`` — see and choose the Project new environments land in."""

from __future__ import annotations

import httpx
import typer

from hud.cli.utils.api import require_api_key
from hud.cli.utils.project import (
    Project,
    ProjectNotFound,
    ProjectNotWritable,
    list_projects,
    report_project_error,
    resolve_placement,
    resolve_project,
)
from hud.cli.utils.source import EnvironmentSource
from hud.utils.exceptions import HudRequestError
from hud.utils.hud_console import HUDConsole
from hud.utils.platform import PlatformClient

project_app = typer.Typer(
    name="project",
    help="Show and choose the HUD Project new environments and tasksets land in",
    add_completion=False,
    rich_markup_mode="rich",
)


@project_app.command("list")
def list_command() -> None:
    """List the Projects you can see.

    [not dim]Examples:
        hud project list[/not dim]
    """
    console = HUDConsole()
    require_api_key("list projects")

    try:
        projects = list_projects(PlatformClient.from_settings())
    except HudRequestError as e:
        raise report_project_error(console, e) from e

    if not projects:
        console.warning("No projects found")
        console.hint("Create one with: hud project create <name>")
        return

    console.info("Your projects:")
    for project in sorted(projects, key=lambda p: (not p.is_default, p.name)):
        tags: list[str] = []
        if project.is_default:
            tags.append("default")
        if not project.can_create:
            tags.append("read-only")
        suffix = f" [{', '.join(tags)}]" if tags else ""
        console.info(f"  {project.name} ({project.short_id}...){suffix}")


@project_app.command("create")
def create_command(
    name: str = typer.Argument(..., help="Name for the new project"),
    description: str | None = typer.Option(None, "--description", help="What the project holds"),
    directory: str = typer.Option(".", "--directory", "-C", help="Directory to pin it to"),
    no_use: bool = typer.Option(False, "--no-use", help="Create without pinning this directory"),
) -> None:
    """Create a Project and pin this directory to it.

    [not dim]Only team admins can create projects.

    Examples:
        hud project create browser-evals
        hud project create browser-evals --no-use[/not dim]
    """
    console = HUDConsole()
    require_api_key("create a project")

    platform = PlatformClient.from_settings()
    payload: dict[str, str] = {"name": name}
    if description:
        payload["description"] = description

    try:
        created = Project.from_record(platform.post("/projects", json=payload))
    except HudRequestError as e:
        if e.status_code == httpx.codes.CONFLICT:
            console.error(f"A project named '{name}' already exists")
            console.hint(f"Pin this directory to it with: hud project use {name}")
            raise typer.Exit(1) from e
        if e.status_code == httpx.codes.FORBIDDEN:
            console.error("Only team admins can create projects")
            raise typer.Exit(1) from e
        raise report_project_error(console, e) from e

    console.success(f"Created project: {created.name} ({created.short_id}...)")
    if not no_use:
        _pin(created, directory, console)


@project_app.command("use")
def use_command(
    ref: str = typer.Argument(..., help="Project name or ID"),
    directory: str = typer.Option(".", "--directory", "-C", help="Directory to pin"),
) -> None:
    """Pin a directory to a Project.

    [not dim]Writes projectId to .hud/config.json, so teammates deploying this
    environment place it in the same project. Set a machine-wide fallback for
    unpinned directories with: hud set HUD_PROJECT=<name>

    Examples:
        hud project use browser-evals
        hud project use browser-evals -C ./envs/browser[/not dim]
    """
    console = HUDConsole()
    require_api_key("select a project")

    try:
        project = resolve_project(PlatformClient.from_settings(), ref)
    except (ProjectNotFound, HudRequestError) as e:
        raise report_project_error(console, e) from e
    if not project.can_create:
        raise report_project_error(console, ProjectNotWritable(project))
    _pin(project, directory, console)


@project_app.callback(invoke_without_command=True)
def project_callback(
    ctx: typer.Context,
    directory: str = typer.Option(".", "--directory", "-C", help="Directory to report on"),
) -> None:
    """Show the Project this directory places new environments and tasksets in.

    [not dim]Examples:
        hud project                      # where does a deploy here land?
        hud project list                 # projects you can see
        hud project use browser-evals    # pin this directory[/not dim]
    """
    if ctx.invoked_subcommand is not None:
        return

    console = HUDConsole()
    require_api_key("resolve the current project")
    try:
        placement = resolve_placement(
            PlatformClient.from_settings(),
            EnvironmentSource.open(directory),
            flag=None,
        )
    except (ProjectNotFound, HudRequestError) as e:
        raise report_project_error(console, e) from e

    console.info(f"Project: {placement.label}")
    if placement.project is None:
        console.hint("Pin a different one with: hud project use <name>")
    elif not placement.project.can_create:
        console.warning("You do not have create access to this Project")


def _pin(project: Project, directory: str, console: HUDConsole) -> None:
    changed = EnvironmentSource.open(directory).save_config({"projectId": project.id})
    console.success(f"Using project: {project.name} ({project.short_id}...)")
    if changed:
        console.dim_info("Config saved to:", ".hud/config.json")


__all__ = ["project_app"]
