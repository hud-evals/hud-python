"""Project lookup and placement resolution for the CLI."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import typer

from hud.utils.exceptions import HudRequestError
from hud.utils.naming import normalize_environment_name

if TYPE_CHECKING:
    from hud.cli.utils.source import EnvironmentSource
    from hud.utils.hud_console import HUDConsole
    from hud.utils.platform import PlatformClient


class ProjectSource(Enum):
    """Where a resolved Project came from, most specific first."""

    FLAG = "--project"
    CONFIG = ".hud/config.json"
    GLOBAL_DEFAULT = "HUD_DEFAULT_PROJECT"
    TEAM_DEFAULT = "team default"


PROJECT_OPTION_HELP = (
    "Project for this command (name or ID). Defaults to the directory's saved "
    "project, HUD_DEFAULT_PROJECT, then your team default. Does not change "
    "directory configuration."
)

_PROJECTS_DISABLED_DETAIL = "projects are not enabled"
_PROJECTS_DISABLED_ERROR = "projects_not_enabled"


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

    @property
    def short_id(self) -> str:
        return self.id[:8]


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

    def __init__(self, ref: str, available: list[Project]) -> None:
        self.ref = ref
        self.available = available
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
    data = platform.get("/projects")
    return _projects_from_page(data)


def require_projects_enabled(platform: PlatformClient) -> None:
    """Check access to the feature-gated Projects API."""
    platform.get("/projects", params={"limit": 1})


def _projects_from_page(data: Any) -> list[Project]:
    """Parse the platform's paginated Project response."""
    records = data.get("items") if isinstance(data, dict) else None
    if not isinstance(records, list):
        return []
    return [Project.from_record(item) for item in records if isinstance(item, dict)]


def resolve_project(platform: PlatformClient, ref: str) -> Project:
    """Map a Project name or id to the Project itself.

    Names are normalized the same way the platform normalizes them on create,
    so `My Project` and `my-project` resolve to the same row.
    """
    try:
        project_id = str(uuid.UUID(ref))
    except ValueError:
        name = normalize_environment_name(ref, default="")
        projects = _projects_from_page(platform.get("/projects", params={"search": name}))
        match = next((p for p in projects if p.name == name), None)
    else:
        try:
            return Project.from_record(platform.get(f"/projects/{project_id}"))
        except HudRequestError as e:
            if e.status_code != 404:
                raise
            match = None

    if match is None:
        raise ProjectNotFound(ref, list_projects(platform))
    return match


def resolve_placement(
    platform: PlatformClient,
    env_source: EnvironmentSource,
    *,
    flag: str | None,
) -> Placement:
    """Resolve the configured Project."""
    from hud.settings import settings

    for ref, source in (
        (flag, ProjectSource.FLAG),
        (env_source.project_id, ProjectSource.CONFIG),
        (settings.default_project, ProjectSource.GLOBAL_DEFAULT),
    ):
        if ref:
            project = resolve_project(platform, ref)
            return Placement(project=project, source=source)

    return Placement(project=None, source=ProjectSource.TEAM_DEFAULT)


def report_project_error(console: HUDConsole, error: Exception) -> typer.Exit:
    """Explain why a Project could not be used, and return the exit to raise."""
    if isinstance(error, HudRequestError) and projects_not_enabled(error):
        console.error("Projects are not enabled for your team")
        console.hint("Contact HUD to enable the Projects beta for your team")
    elif isinstance(error, ProjectNotFound):
        console.error(str(error))
        if error.available:
            console.info("Projects you can see:")
            for candidate in error.available:
                console.info(f"  {candidate.name} ({candidate.short_id}...)")
        else:
            console.hint("Create one with: hud project create <name>")
    elif isinstance(error, ProjectNotWritable):
        console.error(str(error))
        console.hint("Ask a project manager for 'create' scope, or pick another project")
    else:
        console.error(f"Failed to reach the HUD platform: {error}")
    return typer.Exit(1)


def projects_not_enabled(error: HudRequestError) -> bool:
    """Whether the Projects API rejected a caller at its feature gate."""
    if error.status_code != 403 or not isinstance(error.response_json, dict):
        return False
    if error.response_json.get("error") == _PROJECTS_DISABLED_ERROR:
        return True
    detail = error.response_json.get("detail")
    return isinstance(detail, str) and detail.casefold() == _PROJECTS_DISABLED_DETAIL


def resolve_writable_placement(
    platform: PlatformClient,
    env_source: EnvironmentSource,
    *,
    flag: str | None,
    console: HUDConsole,
) -> Placement:
    """Resolve and announce a Project that accepts new resources."""
    placement = resolve_placement_or_exit(platform, env_source, flag=flag, console=console)
    require_writable_placement(placement, console)
    return placement


def resolve_placement_or_exit(
    platform: PlatformClient,
    env_source: EnvironmentSource,
    *,
    flag: str | None,
    console: HUDConsole,
) -> Placement:
    """Resolve and announce a Project without requiring create access."""
    try:
        placement = resolve_placement(platform, env_source, flag=flag)
    except (ProjectNotFound, HudRequestError) as e:
        raise report_project_error(console, e) from e

    console.info(f"Project: {placement.label}")
    return placement


def require_writable_placement(placement: Placement, console: HUDConsole) -> None:
    """Exit when an operation would write to a read-only Project."""
    if placement.project is not None and not placement.project.can_create:
        error = ProjectNotWritable(placement.project)
        raise report_project_error(console, error) from error


__all__ = [
    "PROJECT_OPTION_HELP",
    "Placement",
    "Project",
    "ProjectNotFound",
    "ProjectNotWritable",
    "ProjectSource",
    "list_projects",
    "projects_not_enabled",
    "report_project_error",
    "require_projects_enabled",
    "require_writable_placement",
    "resolve_placement",
    "resolve_placement_or_exit",
    "resolve_project",
    "resolve_writable_placement",
]
