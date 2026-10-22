"""Jira, rebuilt from an issue export: the issue navigator's CSV, or REST API JSON."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Annotated, Any, Self

from fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field

from data_sources.base import (
    DataSource,
    ExportError,
    Limit,
    Page,
    TaskContext,
    counted,
    paginate,
    tool,
)
from data_sources.sources.jql import PRIORITY_RANK, JqlError, Reader, Value, parse
from data_sources.tabular import Columns, tables_with
from data_sources.values import parse_date, parse_datetime, parse_number

_COLUMNS = {
    "key": ("issue key", "key"),
    "summary": ("summary", "title"),
    "type": ("issue type", "type"),
    "status": ("status",),
    "project_key": ("project key",),
    "project_name": ("project name", "project"),
    "priority": ("priority",),
    "resolution": ("resolution",),
    "assignee": ("assignee",),
    "reporter": ("reporter", "creator"),
    "created": ("created",),
    "updated": ("updated",),
    "resolved": ("resolved",),
    "due": ("due date", "due"),
    "labels": ("labels", "label"),
    "sprint": ("sprint",),
    "points": (
        "custom field story points",
        "story points",
        "custom field story point estimate",
        "story point estimate",
    ),
    "parent": ("parent key", "parent", "custom field epic link", "epic link"),
    "description": ("description",),
    "comment": ("comment", "comments"),
}
_DONE = {"done", "closed", "resolved", "complete", "completed", "cancelled", "canceled", "won't do"}
_TO_DO = {"to do", "todo", "open", "backlog", "new", "selected for development", "reopened"}

IssueKey = Annotated[str, Field(description="Issue key, e.g. CLOSE-12")]


class JiraComment(BaseModel):
    author: str | None = None
    created: datetime | None = None
    body: str


class JiraIssueSummary(BaseModel):
    key: str
    summary: str
    status: str
    type: str | None = None
    priority: str | None = None
    assignee: str | None = None
    due: date | None = None
    updated: datetime | None = None


class JiraIssue(JiraIssueSummary):
    project: str
    reporter: str | None = None
    created: datetime | None = None
    resolved: datetime | None = None
    resolution: str | None = None
    labels: list[str] = []
    sprint: str | None = None
    story_points: float | None = None
    parent: str | None = None
    description: str | None = None
    comments: list[JiraComment] = []


class JiraProject(BaseModel):
    key: str
    name: str | None
    issues: int
    open: int
    statuses: dict[str, int]


def status_category(issue: JiraIssue) -> str:
    status = issue.status.lower()
    if issue.resolution or status in _DONE:
        return "Done"
    if status in _TO_DO:
        return "To Do"
    return "In Progress"


class Jira(DataSource):
    id = "jira"
    title = "Jira"
    summary = "Jira issue export (.csv or .json)."

    def __init__(self, context: TaskContext) -> None:
        super().__init__(context)
        self._issues: dict[str, JiraIssue] = {}
        self._project_names: dict[str, str] = {}

    @classmethod
    def load(cls, files: Sequence[Path], context: TaskContext) -> Self:
        jira = cls(context)
        for path in files:
            suffix = path.suffix.lower()
            if suffix == ".json":
                jira._load_json(path)
            else:
                tables = tables_with([path], _COLUMNS, ("key", "summary"))
                if not tables:
                    raise ExportError(f"{path.name}: no Issue key and Summary columns found")
                for columns in tables:
                    for row in columns.table.rows:
                        jira._add_row(columns, row)
        if not jira._issues:
            raise ExportError("No Jira issues found in the uploaded files")
        return jira

    def describe(self) -> str:
        projects = {issue.project for issue in self._issues.values()}
        return f"{counted(len(self._issues), 'issue')} in {counted(len(projects), 'project')}"

    def _add_row(self, columns: Columns, row: Sequence[str]) -> None:
        key = columns.text(row, "key")
        if not key:
            return
        project = columns.text(row, "project_key") or key.split("-")[0]
        if name := columns.text(row, "project_name"):
            self._project_names.setdefault(project, name)
        sprints = columns.texts(row, "sprint")
        resolution = columns.text(row, "resolution")
        self._issues[key] = JiraIssue(
            key=key,
            summary=columns.text(row, "summary") or "",
            status=columns.text(row, "status") or "Open",
            type=columns.text(row, "type"),
            priority=columns.text(row, "priority"),
            assignee=columns.text(row, "assignee"),
            due=columns.date(row, "due"),
            updated=columns.datetime(row, "updated"),
            project=project,
            reporter=columns.text(row, "reporter"),
            created=columns.datetime(row, "created"),
            resolved=columns.datetime(row, "resolved"),
            resolution=None if (resolution or "").lower() == "unresolved" else resolution,
            labels=columns.items(row, "labels"),
            sprint=sprints[-1] if sprints else None,
            story_points=columns.number(row, "points"),
            parent=columns.text(row, "parent"),
            description=columns.text(row, "description"),
            comments=[_csv_comment(text) for text in columns.texts(row, "comment")],
        )

    def _load_json(self, path: Path) -> None:
        try:
            data: Any = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ExportError(f"{path.name}: not valid JSON ({error.msg})") from error
        issues = data.get("issues") if isinstance(data, dict) else data
        if not isinstance(issues, list):
            raise ExportError(f"{path.name}: expected a list of issues or {{'issues': [...]}}")
        for raw in issues:
            if not isinstance(raw, dict):
                continue
            fields: dict[str, Any] = raw.get("fields") or raw
            key = str(raw.get("key") or fields.get("key") or "")
            if not key:
                continue
            project = fields.get("project")
            project_key = _name(project, "key") or key.split("-")[0]
            if project_name := _name(project, "name"):
                self._project_names.setdefault(project_key, project_name)
            comments = fields.get("comment")
            comment_list = (
                comments.get("comments", []) if isinstance(comments, dict) else comments or []
            )
            self._issues[key] = JiraIssue(
                key=key,
                summary=str(fields.get("summary") or ""),
                status=_name(fields.get("status"), "name") or "Open",
                type=_name(fields.get("issuetype") or fields.get("type"), "name"),
                priority=_name(fields.get("priority"), "name"),
                assignee=_name(fields.get("assignee"), "displayName"),
                due=parse_date(fields.get("duedate") or fields.get("due")),
                updated=parse_datetime(fields.get("updated")),
                project=project_key,
                reporter=_name(fields.get("reporter"), "displayName"),
                created=parse_datetime(fields.get("created")),
                resolved=parse_datetime(fields.get("resolutiondate") or fields.get("resolved")),
                resolution=_name(fields.get("resolution"), "name"),
                labels=[str(label) for label in fields.get("labels") or []],
                sprint=_name(fields.get("sprint"), "name"),
                story_points=parse_number(
                    str(fields.get("story_points") or fields.get("storyPoints") or "")
                ),
                parent=_name(fields.get("parent"), "key"),
                description=_document_text(fields.get("description")) or None,
                comments=[
                    JiraComment(
                        author=_name(comment.get("author"), "displayName"),
                        created=parse_datetime(comment.get("created")),
                        body=_document_text(comment.get("body")),
                    )
                    for comment in comment_list
                    if isinstance(comment, dict)
                ],
            )

    def _issue(self, key: str) -> JiraIssue:
        issue = self._issues.get(key.strip().upper())
        if issue is None:
            raise ToolError(f"No issue {key!r}")
        return issue

    def _now(self) -> datetime:
        return datetime.combine(self.context.today, time(12), UTC)

    @tool
    def search_issues(
        self,
        jql: Annotated[
            str,
            Field(
                description=(
                    "JQL, e.g. project = CLOSE AND statusCategory != Done AND due <= endOfMonth()"
                    " ORDER BY priority DESC. Supports = != ~ !~ > >= < <= IN, NOT IN, IS EMPTY,"
                    " AND/OR/NOT, -7d style dates and startOf/endOf Day/Week/Month/Year()."
                )
            ),
        ] = "",
        limit: Limit = 50,
    ) -> Page[JiraIssueSummary]:
        """Search issues with JQL; an empty query lists every issue, most recently updated first."""
        try:
            query = parse(jql, self.context.today)
            found = [issue for issue in self._issues.values() if query.where(_reader(issue))]
        except JqlError as error:
            raise ToolError(f"JQL error: {error}") from error
        for field, descending in reversed(query.order or [("updated", True)]):
            found.sort(key=lambda issue: _sort_key(issue, field, descending), reverse=descending)
        summaries = [JiraIssueSummary.model_validate(issue.model_dump()) for issue in found]
        return paginate(summaries, limit)

    @tool
    def get_issue(self, key: IssueKey) -> JiraIssue:
        """Get an issue with its description and comments."""
        return self._issue(key)

    @tool
    def list_projects(self) -> list[JiraProject]:
        """List projects with issue counts by status."""
        projects: dict[str, list[JiraIssue]] = {}
        for issue in self._issues.values():
            projects.setdefault(issue.project, []).append(issue)
        return [
            JiraProject(
                key=key,
                name=self._project_names.get(key),
                issues=len(issues),
                open=sum(1 for issue in issues if status_category(issue) != "Done"),
                statuses=dict(Counter(issue.status for issue in issues)),
            )
            for key, issues in sorted(projects.items())
        ]

    @tool
    def add_comment(self, key: IssueKey, body: str) -> JiraComment:
        """Comment on an issue."""
        issue = self._issue(key)
        if not body.strip():
            raise ToolError("body is empty")
        comment = JiraComment(author="assistant", created=self._now(), body=body)
        issue.comments.append(comment)
        issue.updated = comment.created
        self.record("add_comment", {"key": issue.key, "body": body}, f"commented on {issue.key}")
        return comment

    @tool
    def transition_issue(
        self,
        key: IssueKey,
        status: Annotated[str, Field(description="Target status, e.g. In Progress or Done")],
    ) -> JiraIssueSummary:
        """Move an issue to another status of its project."""
        issue = self._issue(key)
        statuses = {
            other.status for other in self._issues.values() if other.project == issue.project
        }
        target = next(
            (known for known in statuses if known.lower() == status.strip().lower()), None
        )
        if target is None:
            raise ToolError(
                f"{issue.project} has no status {status!r}; statuses: {', '.join(sorted(statuses))}"
            )
        before = issue.status
        issue.status = target
        issue.updated = self._now()
        if target.lower() in _DONE:
            issue.resolved, issue.resolution = issue.updated, issue.resolution or "Done"
        else:
            issue.resolved, issue.resolution = None, None
        self.record(
            "transition_issue",
            {"key": issue.key, "status": target},
            f"{issue.key}: {before} -> {target}",
        )
        return JiraIssueSummary.model_validate(issue.model_dump())

    @tool
    def create_issue(
        self,
        project: Annotated[str, Field(description="Project key, e.g. CLOSE")],
        summary: str,
        issue_type: str = "Task",
        description: str | None = None,
        assignee: str | None = None,
        priority: str | None = None,
        due: date | None = None,
        labels: list[str] | None = None,
    ) -> JiraIssueSummary:
        """Create an issue in a project."""
        project_key = project.strip().upper()
        numbers = [
            int(issue.key.split("-")[1])
            for issue in self._issues.values()
            if issue.project == project_key and issue.key.split("-")[-1].isdigit()
        ]
        if not numbers:
            known = ", ".join(sorted({issue.project for issue in self._issues.values()}))
            raise ToolError(f"No project {project!r}; projects: {known}")
        key = f"{project_key}-{max(numbers) + 1}"
        if priority and priority.lower() not in PRIORITY_RANK:
            raise ToolError(f"Unknown priority {priority!r}")
        issue = JiraIssue(
            key=key,
            summary=summary,
            status="To Do",
            type=issue_type,
            priority=priority,
            assignee=assignee,
            due=due,
            updated=self._now(),
            project=project_key,
            reporter="assistant",
            created=self._now(),
            labels=labels or [],
            description=description,
        )
        self._issues[key] = issue
        self.record(
            "create_issue",
            {
                "project": project_key,
                "summary": summary,
                "issue_type": issue_type,
                "assignee": assignee,
                "priority": priority,
                "due": due,
                "description": description,
            },
            f"created {key}",
        )
        return JiraIssueSummary.model_validate(issue.model_dump())


def _reader(issue: JiraIssue) -> Reader:
    def read(field: str) -> list[Value]:
        if field in {"created", "updated", "resolved"}:
            moment: datetime | None = getattr(issue, field)
            return [moment.date()] if moment else []
        if field == "due":
            return [issue.due] if issue.due else []
        if field == "storypoints":
            return [issue.story_points] if issue.story_points is not None else []
        if field == "labels":
            return list(issue.labels)
        if field == "comment":
            return [comment.body for comment in issue.comments]
        if field == "text":
            parts = [issue.summary, issue.description or "", *(c.body for c in issue.comments)]
            return [" ".join(parts)]
        if field == "statuscategory":
            return [status_category(issue)]
        value = getattr(issue, field, None)
        return [value] if value else []

    return read


def _sort_key(issue: JiraIssue, field: str, descending: bool) -> tuple[int, Any]:
    """Sort key that keeps issues without a value last in either direction."""
    values = _reader(issue)(field)
    missing = (0, 0) if descending else (1, 0)
    if not values:
        return missing
    present = 1 if descending else 0
    value = values[0]
    if field == "priority":
        return present, PRIORITY_RANK.get(str(value).lower(), 0)
    if field == "key":
        project, _, number = issue.key.partition("-")
        return present, (project, int(number) if number.isdigit() else 0)
    if isinstance(value, date):
        return present, value.toordinal()
    if isinstance(value, float):
        return present, value
    return present, str(value).casefold()


def _csv_comment(text: str) -> JiraComment:
    parts = text.split(";", 2)
    if len(parts) == 3 and (created := parse_datetime(parts[0])):
        return JiraComment(author=parts[1] or None, created=created, body=parts[2].strip())
    return JiraComment(body=text.strip())


def _name(value: Any, attribute: str) -> str | None:
    if isinstance(value, dict):
        found = value.get(attribute) or value.get("name") or value.get("displayName")
        return str(found) if found else None
    return str(value) if value else None


def _document_text(value: Any) -> str:
    """Plain text from a string or an Atlassian Document Format node."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if value.get("type") == "text":
            return str(value.get("text", ""))
        children = "".join(_document_text(child) for child in value.get("content", []))
        return children + (
            "\n" if value.get("type") in {"paragraph", "heading", "listItem"} else ""
        )
    if isinstance(value, list):
        return "".join(_document_text(child) for child in value)
    return str(value)
