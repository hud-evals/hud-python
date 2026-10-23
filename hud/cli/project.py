"""Projects: platform records, placement, and the ``hud project`` commands."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import typer
from pydantic import AliasPath, BaseModel, ConfigDict, Field

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
from hud.utils.naming import normalize_environment_name
from hud.utils.platform import PlatformClient

PROJECT_OPTION_HELP = (
    "Project name or ID for this command. Defaults to the directory's saved "
    "project, HUD_DEFAULT_PROJECT, then your team default. Does not change "
    "directory configuration."
)


class Project(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: uuid.UUID
    name: str
    is_default: bool = False
    can_create: bool = Field(default=False, validation_alias=AliasPath("capabilities", "create"))

    @classmethod
    def resolve(cls, platform: PlatformClient, ref: str) -> Project:
        """The Project with this ID or name, within the authenticated scope.

        Names are normalized the way the platform normalizes them on create, so
        ``My Project`` and ``my-project`` resolve to the same row.
        """
        try:
            project_id = uuid.UUID(ref)
        except ValueError:
            project_id = None
        if project_id is None:
            name = normalize_environment_name(ref, default="")
            found = platform.get("/projects", _ProjectPage, params={"search": name, "limit": 500})
            matches = [project for project in found.items if project.name == name]
            if not matches:
                raise CliError(
                    "not_found",
                    f"No Project named {ref!r}.",
                    suggestion="Run 'hud project list' to see visible Projects.",
                    input={"project": ref},
                )
            if len(matches) > 1:
                raise CliError(
                    "usage",
                    f"{len(matches)} Projects are named {name!r}; pass an ID instead: "
                    + ", ".join(str(project.id) for project in matches),
                    input={"project": ref},
                )
            return matches[0]
        try:
            return platform.get(f"/projects/{project_id}", cls)
        except HudRequestError as exc:
            raise CliError.from_http(exc, resource="Project", input={"project": ref}) from exc


class _ProjectPage(BaseModel):
    items: list[Project]
    total: int


@dataclass(frozen=True)
class Placement:
    """The Project selected for the current directory, and what selected it."""

    project: Project | None
    #: ``--project``, ``.hud/config.json``, ``HUD_DEFAULT_PROJECT``, or ``team default``.
    source: str

    @property
    def project_id(self) -> str | None:
        """The id to send to the platform, or None to accept the team default."""
        return str(self.project.id) if self.project else None

    @property
    def label(self) -> str:
        if self.project is None:
            return "team default Project"
        return f"{self.project.name} (via {self.source})"

    @classmethod
    def resolve(
        cls, platform: PlatformClient, link: DirectoryLink, *, flag: str | None
    ) -> Placement:
        """The configured Project, most specific source first."""
        for ref, source in (
            (flag, "--project"),
            (str(link.project_id) if link.project_id else None, str(CONFIG_PATH)),
            (settings.default_project, "HUD_DEFAULT_PROJECT"),
        ):
            if ref:
                return cls(Project.resolve(platform, ref), source)
        return cls(None, "team default")

    def require_writable(self) -> None:
        if self.project is not None and not self.project.can_create:
            raise CliError(
                error="permission_denied",
                message="You do not have permission to create environments or tasksets in "
                f"project '{self.project.name}'",
                input={"project": str(self.project.id)},
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
    platform = PlatformClient.from_settings()
    projects: list[Project] = []
    while True:
        page = platform.get(
            "/projects", _ProjectPage, params={"limit": 500, "offset": len(projects)}
        )
        projects.extend(page.items)
        if len(projects) >= page.total:
            break
        if not page.items:
            raise HudRequestError("Projects API returned an empty page before the reported total")
    if quiet:
        for project in projects:
            typer.echo(project.id)
        return [project.model_dump(mode="json") for project in projects]
    console = HUDConsole()
    for project in projects:
        tags = " (default)" if project.is_default else ""
        tags += " (read-only)" if not project.can_create else ""
        console.info(f"{project.name}  {project.id}{tags}")
    if not projects:
        console.info("No projects found")
    return [project.model_dump(mode="json") for project in projects]


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
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompts (required in non-interactive terminals).",
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
    CLI.confirm_or_abort(f"Create Project {name}?", yes=yes, default=True)
    created = platform.post("/projects", Project, json=payload)
    if state is not None:
        state.update(DirectoryLink(project_id=created.id))
    HUDConsole().success(f"Created Project: {created.name} ({created.id})")
    return created.model_dump(mode="json")


@project_app.command("use")
def use_command(
    ctx: typer.Context,
    ref: str = typer.Argument(..., help="Project name or ID (see hud project list)"),
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
    placement = Placement(Project.resolve(platform, ref), "--project")
    placement.require_writable()
    project = placement.project
    assert project is not None
    if not dry_run:
        state.update(DirectoryLink(project_id=project.id))
    HUDConsole().success(
        f"{'Would use' if dry_run else 'Using'} Project: {project.name} ({project.id})"
    )
    return {**project.model_dump(mode="json"), "dry_run": dry_run}


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
    platform.get("/projects", _ProjectPage, params={"limit": 1})
    state = DirectoryState(AuthScope.resolve(platform), directory)
    placement = Placement.resolve(platform, state.load(), flag=None)
    HUDConsole().info(f"Project: {placement.label}")
    return {
        "project": placement.project.model_dump(mode="json") if placement.project else None,
        "source": placement.source,
        "label": placement.label,
    }
