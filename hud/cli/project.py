"""Projects: platform records, placement, and the ``hud project`` commands."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from typing import Any

import typer

from hud.cli import (
    CLI,
    CONFIG_PATH,
    AuthScope,
    CliError,
    DirectoryLink,
    DirectoryState,
)
from hud.settings import settings
from hud.utils.exceptions import HudRequestError
from hud.utils.hud_console import HUDConsole
from hud.utils.platform import PlatformClient

PROJECT_OPTION_HELP = (
    "Project ID for this command. Defaults to the directory's saved "
    "project, HUD_DEFAULT_PROJECT, then your team default. Does not change "
    "directory configuration."
)


@dataclass(frozen=True)
class Project:
    id: str
    name: str
    is_default: bool
    can_create: bool

    @classmethod
    def from_record(cls, data: dict[str, Any]) -> Project:
        return cls(
            id=str(data["id"]),
            name=data["name"],
            is_default=bool(data.get("is_default")),
            can_create=bool(data.get("capabilities", {}).get("create")),
        )


@dataclass(frozen=True)
class Placement:
    """The Project selected for the current directory, and what selected it."""

    project: Project | None
    #: ``--project``, ``.hud/config.json``, ``HUD_DEFAULT_PROJECT``, or ``team default``.
    source: str

    @property
    def project_id(self) -> str | None:
        """The id to send to the platform, or None to accept the team default."""
        return self.project.id if self.project else None

    @property
    def label(self) -> str:
        if self.project is None:
            return "team default Project"
        return f"{self.project.name} (via {self.source})"


def list_projects(platform: PlatformClient) -> list[Project]:
    """Every Project visible to the caller."""
    projects: list[Project] = []
    while True:
        data = platform.get("/projects", params={"limit": 500, "offset": len(projects)})
        page = [Project.from_record(item) for item in data["items"]]
        projects.extend(page)
        if len(projects) >= data["total"]:
            return projects
        if not page:
            raise ValueError("Projects API returned an empty page before the reported total")


def resolve_project(platform: PlatformClient, ref: str) -> Project:
    """The Project with this canonical ID, within the authenticated scope."""
    try:
        project_id = str(uuid.UUID(ref))
    except ValueError as exc:
        raise ValueError(
            "Pass a Project ID from 'hud project list'; name lookup is not supported"
        ) from exc
    try:
        return Project.from_record(platform.get(f"/projects/{project_id}"))
    except HudRequestError as exc:
        raise CliError.from_http(exc, resource="Project", input={"project": ref}) from exc


def resolve_placement(
    platform: PlatformClient, link: DirectoryLink, *, flag: str | None
) -> Placement:
    """The configured Project, most specific source first."""
    for ref, source in (
        (flag, "--project"),
        (str(link.project_id) if link.project_id else None, str(CONFIG_PATH)),
        (settings.default_project, "HUD_DEFAULT_PROJECT"),
    ):
        if ref:
            return Placement(resolve_project(platform, ref), source)
    return Placement(None, "team default")


def require_writable_placement(placement: Placement) -> None:
    if placement.project is not None and not placement.project.can_create:
        raise CliError(
            error="permission_denied",
            message="You do not have permission to create environments or tasksets in "
            f"project '{placement.project.name}'",
            input={"project": placement.project.id},
        )


project_app = CLI(
    name="project",
    help="Show and choose the Project for new environments and tasksets",
    add_completion=False,
    rich_markup_mode="rich",
)


@project_app.command("list")
def list_command(
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Print one identifier per line, with no headers (for piping)."
    ),
) -> Any:
    """List all visible Projects and their canonical IDs."""
    projects = list_projects(PlatformClient.from_settings())
    if quiet:
        for project in projects:
            typer.echo(project.id)
        return [asdict(project) for project in projects]
    console = HUDConsole()
    for project in projects:
        tags = " (default)" if project.is_default else ""
        tags += " (read-only)" if not project.can_create else ""
        console.info(f"{project.name}  {project.id}{tags}")
    if not projects:
        console.info("No projects found")
    return [asdict(project) for project in projects]


@project_app.command("create")
def create_command(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Name for the new Project"),
    description: str | None = typer.Option(None, "--description"),
    directory: str | None = typer.Option(None, "--directory", "-C"),
    no_use: bool = typer.Option(False, "--no-use", help="Create without linking this directory"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the planned action without making changes."
    ),
) -> Any:
    """Create a Project and link this directory unless --no-use is passed."""
    platform = PlatformClient.from_settings()
    payload = {"name": name}
    if description:
        payload["description"] = description
    if dry_run:
        HUDConsole().info(f"Would create Project {name}")
        return {"dry_run": True, "action": "create_project", **payload}
    state = (
        None
        if no_use
        else DirectoryState(
            AuthScope.resolve(platform), directory or ctx.meta["hud_project_directory"]
        )
    )
    if state is not None:
        state.load()  # a config written under other credentials fails before we create anything
    created = Project.from_record(platform.post("/projects", json=payload))
    if state is not None:
        state.update(DirectoryLink(project_id=created.id))
    HUDConsole().success(f"Created Project: {created.name} ({created.id})")
    return asdict(created)


@project_app.command("use")
def use_command(
    ctx: typer.Context,
    ref: str = typer.Argument(..., help="Project ID from hud project list"),
    directory: str | None = typer.Option(None, "--directory", "-C"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the planned action without making changes."
    ),
) -> Any:
    """Link a directory to a Project in .hud/config.json for this account and team."""
    platform = PlatformClient.from_settings()
    state = DirectoryState(
        AuthScope.resolve(platform), directory or ctx.meta["hud_project_directory"]
    )
    state.load()
    project = resolve_project(platform, ref)
    require_writable_placement(Placement(project, "--project"))
    if not dry_run:
        state.update(DirectoryLink(project_id=project.id))
    HUDConsole().success(
        f"{'Would use' if dry_run else 'Using'} Project: {project.name} ({project.id})"
    )
    return {**asdict(project), "dry_run": dry_run}


@project_app.callback(invoke_without_command=True)
def project_callback(
    ctx: typer.Context,
    directory: str = typer.Option(".", "--directory", "-C"),
) -> Any:
    """Show the Project selected for this directory."""
    ctx.meta["hud_project_directory"] = directory
    if ctx.invoked_subcommand is not None:
        return
    platform = PlatformClient.from_settings()
    # Projects are feature-gated per team; surface that before reading the directory.
    platform.get("/projects", params={"limit": 1})
    state = DirectoryState(AuthScope.resolve(platform), directory)
    placement = resolve_placement(platform, state.load(), flag=None)
    HUDConsole().info(f"Project: {placement.label}")
    return {
        "project": asdict(placement.project) if placement.project else None,
        "source": placement.source,
        "label": placement.label,
    }
