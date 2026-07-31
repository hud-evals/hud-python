"""A minimal mock GitHub for SDLC tasks.

One in-process store holds issues (seeded per task) and the pull requests and
comments the agent creates; :func:`serve` publishes it as ``github_*`` MCP
tools over an ``mcp`` capability. The remote itself is a bare git repo on
disk (see :func:`coding.repo.create_remote`) — the agent pushes branches with
plain git and references them from pull requests here.
"""

from __future__ import annotations

import asyncio
import json
import socket
from dataclasses import asdict, dataclass, field
from typing import Any

from fastmcp import FastMCP
from hud.capabilities import Capability


@dataclass
class Issue:
    number: int
    title: str
    body: str
    labels: list[str] = field(default_factory=list)
    state: str = "open"
    comments: list[str] = field(default_factory=list)


@dataclass
class PullRequest:
    number: int
    title: str
    body: str
    head: str
    base: str
    state: str = "open"
    comments: list[str] = field(default_factory=list)


class MockGitHub:
    """Issue/PR store with an action log (for rubric grading)."""

    def __init__(self) -> None:
        self.issues: dict[int, Issue] = {}
        self.pull_requests: dict[int, PullRequest] = {}
        self.actions: list[dict[str, Any]] = []

    def seed(self, issues: list[dict[str, Any]]) -> None:
        """Reset the store to the task's fixtures."""
        self.issues = {
            int(entry["number"]): Issue(
                number=int(entry["number"]),
                title=str(entry["title"]),
                body=str(entry.get("body", "")),
                labels=[str(label) for label in entry.get("labels", [])],
            )
            for entry in issues
        }
        self.pull_requests = {}
        self.actions = []

    def _record(self, action: str, **details: Any) -> None:
        self.actions.append({"action": action, **details})

    # ─── operations (the tool surface) ────────────────────────────────

    def list_issues(self) -> list[dict[str, Any]]:
        return [asdict(issue) for issue in self.issues.values()]

    def get_issue(self, number: int) -> dict[str, Any]:
        return asdict(self._issue(number))

    def close_issue(self, number: int) -> dict[str, Any]:
        issue = self._issue(number)
        issue.state = "closed"
        self._record("close_issue", number=number)
        return asdict(issue)

    def comment_on_issue(self, number: int, body: str) -> dict[str, Any]:
        issue = self._issue(number)
        issue.comments.append(body)
        self._record("comment_on_issue", number=number, body=body)
        return asdict(issue)

    def create_pull_request(self, title: str, body: str, head: str, base: str) -> dict[str, Any]:
        number = len(self.pull_requests) + 1
        pr = PullRequest(number=number, title=title, body=body, head=head, base=base)
        self.pull_requests[number] = pr
        self._record("create_pull_request", number=number, title=title, head=head)
        return asdict(pr)

    def list_pull_requests(self) -> list[dict[str, Any]]:
        return [asdict(pr) for pr in self.pull_requests.values()]

    def _issue(self, number: int) -> Issue:
        if number not in self.issues:
            raise ValueError(f"no such issue: #{number}")
        return self.issues[number]

    # ─── grading views ────────────────────────────────────────────────

    def latest_pull_request(self) -> PullRequest | None:
        return next(reversed(self.pull_requests.values()), None)

    def transcript(self) -> str:
        """Everything the agent did here, for rubric judging."""
        return json.dumps(
            {
                "issues": [asdict(issue) for issue in self.issues.values()],
                "pull_requests": [asdict(pr) for pr in self.pull_requests.values()],
                "actions": self.actions,
            },
            indent=2,
        )


def serve(github: MockGitHub) -> tuple[asyncio.Task[None], Capability]:
    """Serve *github* as ``github_*`` MCP tools; returns the server task + capability."""
    server: FastMCP = FastMCP(name="github")

    @server.tool
    def github_list_issues() -> list[dict[str, Any]]:
        """List all issues in the repository."""
        return github.list_issues()

    @server.tool
    def github_get_issue(number: int) -> dict[str, Any]:
        """Get one issue by number."""
        return github.get_issue(number)

    @server.tool
    def github_close_issue(number: int) -> dict[str, Any]:
        """Close an issue."""
        return github.close_issue(number)

    @server.tool
    def github_comment_on_issue(number: int, body: str) -> dict[str, Any]:
        """Add a comment to an issue."""
        return github.comment_on_issue(number, body)

    @server.tool
    def github_create_pull_request(title: str, body: str, head: str, base: str = "main") -> dict[str, Any]:
        """Open a pull request from branch *head* (already pushed to the remote) into *base*."""
        return github.create_pull_request(title, body, head, base)

    @server.tool
    def github_list_pull_requests() -> list[dict[str, Any]]:
        """List all pull requests."""
        return github.list_pull_requests()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    task = asyncio.create_task(server.run_async(transport="http", host="127.0.0.1", port=port, show_banner=False))
    return task, Capability.mcp(name="github", url=f"http://127.0.0.1:{port}/mcp")
