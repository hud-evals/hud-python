"""Projects: platform records, placement, and the ``hud project`` commands."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import typer

from hud.cli import require_api_key
from hud.cli.config import CONFIG_PATH, AuthScope, DirectoryLink, DirectoryState
from hud.cli.io import (
    json_option,
    report,
)
from hud.settings import settings
from hud.utils.exceptions import HudRequestError
from hud.utils.hud_console import HUDConsole
from hud.utils.platform import PlatformClient


class ProjectSource(Enum):
    """Where a resolved Project came from, most specific first."""

    FLAG = "--project"
    CONFIG = str(CONFIG_PATH)
    GLOBAL_DEFAULT = "HUD_DEFAULT_PROJECT"
    TEAM_DEFAULT = "team default"


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
        capabilities = data.get("capabilities")
        return cls(
            id=str(data["id"]),
            name=str(data.get("name") or "unnamed"),
            is_default=bool(data.get("is_default")),
            can_create=bool(capabilities.get("create"))
            if isinstance(capabilities, dict)
            else False,
        )


@dataclass(frozen=True)
class Placement:
    """The Project selected for the current directory."""

    project: Project | None
    source: ProjectSource

    @property
    def project_id(self) -> str | None:
        """The id to send to the platform, or None to accept the team default."""
        return self.project.id if self.project else None

    @property
    def label(self) -> str:
        if self.project is None:
            return "team default Project"
        return f"{self.project.name} (via {self.source.value})"


class ProjectNotFound(LookupError):
    """No visible Project matches the given reference."""

    def __init__(self, ref: str) -> None:
        self.ref = ref
        super().__init__(f"No project found matching '{ref}'")


class ProjectNotWritable(PermissionError):
    """The caller may see the Project but may not create resources in it."""

    def __init__(self, project: Project) -> None:
        self.project = project
        super().__init__(
            f"You do not have permission to create environments or tasksets in "
            f"project '{project.name}'"
        )


def list_projects(platform: PlatformClient) -> list[Project]:
    """Every Project visible to the caller."""
    projects: list[Project] = []
    offset = 0
    while True:
        data = platform.get("/projects", params={"limit": 500, "offset": offset})
        page = _projects_from_page(data)
        projects.extend(page)
        offset += len(page)
        if offset >= data["total"]:
            return projects
        if not page:
            raise ValueError("Projects API returned an empty page before the reported total")


def _projects_from_page(data: Any) -> list[Project]:
    records = data.get("items") if isinstance(data, dict) else None
    if not isinstance(records, list):
        return []
    return [Project.from_record(item) for item in records if isinstance(item, dict)]


def resolve_project(platform: PlatformClient, ref: str) -> Project:
    """Resolve a canonical Project ID within the authenticated scope."""
    try:
        project_id = str(uuid.UUID(ref))
    except ValueError as exc:
        raise ValueError(
            "Pass a Project ID from 'hud project list'; name lookup is not supported"
        ) from exc
    try:
        return Project.from_record(platform.get(f"/projects/{project_id}"))
    except HudRequestError as exc:
        if exc.status_code != 404:
            raise
        raise ProjectNotFound(ref) from exc


def resolve_placement(
    platform: PlatformClient,
    link: DirectoryLink,
    *,
    flag: str | None,
) -> Placement:
    """Resolve the configured Project."""
    for ref, source in (
        (flag, ProjectSource.FLAG),
        (str(link.project_id) if link.project_id else None, ProjectSource.CONFIG),
        (settings.default_project, ProjectSource.GLOBAL_DEFAULT),
    ):
        if ref:
            project = resolve_project(platform, ref)
            return Placement(project=project, source=source)

    return Placement(project=None, source=ProjectSource.TEAM_DEFAULT)


def require_writable_placement(placement: Placement) -> None:
    if placement.project is not None and not placement.project.can_create:
        raise ProjectNotWritable(placement.project)


project_app = typer.Typer(
    name="project",
    help="Show and choose the Project for new environments and tasksets",
    add_completion=False,
    rich_markup_mode="rich",
)


@project_app.command("list")
def list_command(
    json_output: bool = json_option(),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Print one identifier per line, with no headers (for piping)."
    ),
) -> None:
    """List all visible Projects and their canonical IDs."""
    require_api_key("list projects")
    rows = [asdict(project) for project in list_projects(PlatformClient.from_settings())]

    def _render(projects: list[dict[str, Any]]) -> None:
        console = HUDConsole()
        for project in projects:
            tags = " (default)" if project["is_default"] else ""
            tags += " (read-only)" if not project["can_create"] else ""
            console.info(f"{project['name']}  {project['id']}{tags}")
        if not projects:
            console.info("No projects found")

    report(
        rows,
        json_output=json_output,
        quiet=quiet,
        ids=lambda projects: [project["id"] for project in projects],
        render=_render,
    )


@project_app.command("create")
def create_command(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Name for the new Project"),
    description: str | None = typer.Option(None, "--description"),
    directory: str | None = typer.Option(None, "--directory", "-C"),
    no_use: bool = typer.Option(False, "--no-use", help="Create without linking this directory"),
    json_output: bool = json_option(),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the planned action without making changes."
    ),
) -> None:
    """Create a Project and link this directory unless --no-use is passed."""
    require_api_key("create a project")
    platform = PlatformClient.from_settings()
    payload = {"name": name}
    if description:
        payload["description"] = description
    if dry_run:
        plan = {"dry_run": True, "action": "create_project", **payload}
        report(
            plan,
            json_output=json_output,
            render=lambda saved: HUDConsole().info(f"Would create Project {saved['name']}"),
        )
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
    saved = asdict(created)
    report(
        saved,
        json_output=json_output,
        render=lambda row: HUDConsole().success(f"Created Project: {row['name']} ({row['id']})"),
    )


@project_app.command("use")
def use_command(
    ctx: typer.Context,
    ref: str = typer.Argument(..., help="Project ID from hud project list"),
    directory: str | None = typer.Option(None, "--directory", "-C"),
    json_output: bool = json_option(),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the planned action without making changes."
    ),
) -> None:
    """Link a directory to a Project in .hud/config.json for this account and team."""
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
    saved = {**asdict(project), "dry_run": dry_run}
    report(
        saved,
        json_output=json_output,
        render=lambda row: HUDConsole().success(
            f"{'Would use' if row['dry_run'] else 'Using'} Project: {row['name']} ({row['id']})"
        ),
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
    saved = {
        "project": asdict(placement.project) if placement.project else None,
        "source": placement.source.value,
        "label": placement.label,
    }
    report(
        saved,
        json_output=json_output,
        render=lambda row: HUDConsole().info(f"Project: {row['label']}"),
    )
