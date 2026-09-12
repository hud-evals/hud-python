"""Project lookup and placement resolution for the CLI."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from hud.utils.exceptions import HudRequestError

if TYPE_CHECKING:
    from hud.cli.utils.config import DirectoryLink
    from hud.utils.platform import PlatformClient


class ProjectSource(Enum):
    """Where a resolved Project came from, most specific first."""

    FLAG = "--project"
    CONFIG = "~/.hud/config.json"
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
    """Parse the platform's paginated Project response."""
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
    from hud.settings import settings

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
