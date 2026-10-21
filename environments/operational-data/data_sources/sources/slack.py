"""Slack, rebuilt from a workspace export (.zip) or channel history files (.json)."""

from __future__ import annotations

import json
import re
import zipfile
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, Self

from fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field

from data_sources.base import (
    DataSource,
    ExportError,
    Limit,
    Page,
    TaskContext,
    contains_all,
    counted,
    paginate,
    resolve,
    search_terms,
    tool,
)

_SKIPPED_SUBTYPES = frozenset(
    {"channel_join", "channel_leave", "channel_purpose", "channel_topic", "channel_name", "bot_add"}
)
_LINK = re.compile(r"<([^<>|]+)(?:\|([^<>]+))?>")
_CHANNEL_FILES: tuple[tuple[str, Literal["public", "private", "direct"]], ...] = (
    ("channels.json", "public"),
    ("groups.json", "private"),
    ("mpims.json", "private"),
    ("dms.json", "direct"),
)

ChannelName = Annotated[str, Field(description="Channel name, e.g. #finance-close")]


class SlackUser(BaseModel):
    id: str
    handle: str
    name: str
    title: str | None = None
    email: str | None = None
    timezone: str | None = None
    is_bot: bool = False


class SlackChannel(BaseModel):
    id: str
    name: str
    kind: Literal["public", "private", "direct"]
    purpose: str | None = None
    topic: str | None = None
    members: list[str] = []
    message_count: int = 0


class SlackMessage(BaseModel):
    channel: str
    ts: str
    time: datetime
    author: str
    text: str
    thread_ts: str | None = None
    reply_count: int = 0
    reactions: list[str] = []
    files: list[str] = []


class Slack(DataSource):
    id = "slack"
    title = "Slack"
    summary = "Slack workspace export (.zip or .json)."

    def __init__(self, context: TaskContext) -> None:
        super().__init__(context)
        self._users: dict[str, SlackUser] = {}
        self._channels: dict[str, SlackChannel] = {}
        self._channel_ids: dict[str, str] = {}
        self._raw: dict[str, list[dict[str, Any]]] = {}
        self._messages: dict[str, list[SlackMessage]] = {}

    @classmethod
    def load(cls, files: Sequence[Path], context: TaskContext) -> Self:
        slack = cls(context)
        for path in files:
            suffix = path.suffix.lower()
            if suffix == ".zip":
                slack._load_zip(path)
            elif suffix == ".json":
                slack._load_json(path)
            else:
                raise ExportError(f"{path.name}: expected a Slack export .zip or channel .json")
        slack._render()
        return slack

    def describe(self) -> str:
        count = sum(len(messages) for messages in self._messages.values())
        return (
            f"{counted(len(self._channels), 'channel')}, "
            f"{counted(len(self._users), 'person', 'people')}, {counted(count, 'message')}"
        )

    def _load_zip(self, path: Path) -> None:
        try:
            archive = zipfile.ZipFile(path)
        except zipfile.BadZipFile as error:
            raise ExportError(f"{path.name}: not a zip archive") from error
        with archive:
            members = [PurePosixPath(name) for name in archive.namelist() if name.endswith(".json")]
            markers = [m.parent for m in members if m.name in {"users.json", "channels.json"}]
            root = min(markers, key=lambda parent: len(parent.parts), default=PurePosixPath("."))

            def read(member: PurePosixPath) -> Any:
                return json.loads(archive.read(str(member)).decode("utf-8"))

            if (users := root / "users.json") in members:
                for raw in read(users):
                    self._add_user(raw)
            folders: dict[str, str] = {}
            for filename, kind in _CHANNEL_FILES:
                if (listing := root / filename) in members:
                    for raw in read(listing):
                        name = self._add_channel(raw, kind)
                        folders[str(raw.get("name") or raw.get("id"))] = name
            for member in members:
                if member.parent.parent != root or member.parent == root:
                    continue
                folder = member.parent.name
                channel = folders.get(folder) or self._add_channel({"name": folder}, "public")
                self._raw.setdefault(channel, []).extend(read(member))
        if not self._raw:
            raise ExportError(f"{path.name}: no channel messages found in this Slack export")

    def _load_json(self, path: Path) -> None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ExportError(f"{path.name}: not valid JSON ({error.msg})") from error
        if isinstance(data, dict) and isinstance(data.get("messages"), list):
            for raw in data.get("users", []):
                self._add_user(raw)
            name = self._add_channel(data.get("channel") or {"name": path.stem}, "public")
            self._raw.setdefault(name, []).extend(data["messages"])
        elif isinstance(data, list):
            name = self._add_channel({"name": path.stem}, "public")
            self._raw.setdefault(name, []).extend(data)
        else:
            raise ExportError(f"{path.name}: expected a list of Slack messages")

    def _add_user(self, raw: dict[str, Any]) -> None:
        profile = raw.get("profile", {})
        user_id = str(raw.get("id", ""))
        if not user_id:
            return
        handle = str(raw.get("name") or user_id)
        self._users[user_id] = SlackUser(
            id=user_id,
            handle=handle,
            name=str(raw.get("real_name") or profile.get("real_name") or handle),
            title=profile.get("title") or None,
            email=profile.get("email") or None,
            timezone=raw.get("tz") or None,
            is_bot=bool(raw.get("is_bot")),
        )

    def _add_channel(
        self, raw: dict[str, Any] | str, kind: Literal["public", "private", "direct"]
    ) -> str:
        if isinstance(raw, str):
            raw = {"name": raw}
        channel_id = str(raw.get("id") or raw.get("name"))
        members = [str(member) for member in raw.get("members", [])]
        name = str(raw.get("name") or "")
        if kind == "direct" or not name:
            handles = sorted(self._display(member, handle=True) for member in members)
            name = "dm-" + "-".join(handles) if handles else channel_id
        name = name.lstrip("#")
        if name not in self._channels:
            self._channels[name] = SlackChannel(
                id=channel_id,
                name=name,
                kind=kind,
                purpose=(raw.get("purpose") or {}).get("value") or None,
                topic=(raw.get("topic") or {}).get("value") or None,
                members=members,
            )
            self._channel_ids[channel_id] = name
        return name

    def _display(self, user_id: str, *, handle: bool = False) -> str:
        user = self._users.get(user_id)
        if user is None:
            return user_id
        return user.handle if handle else user.name

    def _render(self) -> None:
        for raws in self._raw.values():
            for raw in raws:
                profile = raw.get("user_profile")
                user_id = str(raw.get("user", ""))
                if profile and user_id and user_id not in self._users:
                    self._add_user({"id": user_id, "name": profile.get("name"), "profile": profile})
        for name, raws in self._raw.items():
            channel = self._channels[name]
            channel.members = [self._display(member) for member in channel.members]
            messages = [m for raw in raws if (m := self._message(name, raw)) is not None]
            messages.sort(key=lambda message: float(message.ts))
            replies: dict[str, int] = {}
            for message in messages:
                if message.thread_ts and message.thread_ts != message.ts:
                    replies[message.thread_ts] = replies.get(message.thread_ts, 0) + 1
            for message in messages:
                message.reply_count = max(message.reply_count, replies.get(message.ts, 0))
            self._messages[name] = messages
            channel.message_count = len(messages)
        self._raw.clear()

    def _message(self, channel: str, raw: dict[str, Any]) -> SlackMessage | None:
        if raw.get("subtype") in _SKIPPED_SUBTYPES:
            return None
        ts = str(raw.get("ts", ""))
        try:
            when = datetime.fromtimestamp(float(ts), UTC)
        except ValueError:
            return None
        user_id = str(raw.get("user", ""))
        author = (
            self._display(user_id)
            if user_id
            else str(raw.get("username") or (raw.get("bot_profile") or {}).get("name") or "bot")
        )
        thread_ts = raw.get("thread_ts")
        return SlackMessage(
            channel=channel,
            ts=ts,
            time=when,
            author=author,
            text=self._text(str(raw.get("text", ""))),
            thread_ts=str(thread_ts) if thread_ts else None,
            reply_count=int(raw.get("reply_count") or 0),
            reactions=[
                f":{reaction['name']}: x{reaction.get('count', len(reaction.get('users', [])))}"
                for reaction in raw.get("reactions", [])
                if reaction.get("name")
            ],
            files=[
                str(file.get("name") or file.get("title"))
                for file in raw.get("files", [])
                if file.get("name") or file.get("title")
            ],
        )

    def _text(self, text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            target, label = match.group(1), match.group(2)
            if target.startswith("@"):
                return "@" + self._display(target[1:])
            if target.startswith("#"):
                return "#" + (label or self._channel_ids.get(target[1:], target[1:]))
            if target.startswith("!"):
                return label or "@" + target[1:].split("^")[0]
            return f"{label} ({target})" if label and label != target else target

        rendered = _LINK.sub(replace, text)
        return rendered.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")

    def _channel(self, channel: str) -> str:
        return resolve(channel, {name: name for name in self._channels}, "channel")

    def _author_matches(self, wanted: str, author: str) -> bool:
        needle = wanted.lstrip("@").lower()
        if needle in author.lower():
            return True
        return any(
            user.name == author and needle in (user.handle.lower(), (user.email or "").lower())
            for user in self._users.values()
        )

    @tool
    def list_channels(self) -> list[SlackChannel]:
        """List the workspace's channels and direct messages with purpose, topic and members."""
        return sorted(self._channels.values(), key=lambda channel: (channel.kind, channel.name))

    @tool
    def read_channel(
        self,
        channel: ChannelName,
        oldest: datetime | None = None,
        latest: datetime | None = None,
        limit: Limit = 50,
    ) -> Page[SlackMessage]:
        """Read a channel in time order: its most recent messages between oldest and latest.

        Thread replies are left out; a message with reply_count > 0 has a thread
        you can open with read_thread.
        """
        name = self._channel(channel)
        selected = [
            message
            for message in self._messages.get(name, [])
            if message.thread_ts in (None, message.ts)
            and (oldest is None or message.time >= _aware(oldest))
            and (latest is None or message.time <= _aware(latest))
        ]
        return Page(total=len(selected), items=selected[-limit:])

    @tool
    def read_thread(
        self,
        channel: ChannelName,
        thread_ts: Annotated[str, Field(description="ts of the thread's parent message")],
    ) -> list[SlackMessage]:
        """Read a thread: the parent message followed by every reply."""
        name = self._channel(channel)
        thread = [
            message
            for message in self._messages.get(name, [])
            if message.ts == thread_ts or message.thread_ts == thread_ts
        ]
        if not thread:
            raise ToolError(f"No thread {thread_ts} in #{name}")
        return thread

    @tool
    def search_messages(
        self,
        query: Annotated[
            str, Field(description='Words to find; quote a phrase to match it exactly: "wire fee"')
        ],
        channel: ChannelName | None = None,
        author: Annotated[str | None, Field(description="Name or handle of the author")] = None,
        after: date | None = None,
        before: date | None = None,
        limit: Limit = 20,
    ) -> Page[SlackMessage]:
        """Search messages, thread replies included, across channels; newest first."""
        terms = search_terms(query)
        names = [self._channel(channel)] if channel else list(self._messages)
        hits = [
            message
            for name in names
            for message in self._messages.get(name, [])
            if contains_all(terms, message.text, " ".join(message.files))
            and (author is None or self._author_matches(author, message.author))
            and (after is None or message.time.date() >= after)
            and (before is None or message.time.date() <= before)
        ]
        hits.sort(key=lambda message: float(message.ts), reverse=True)
        return paginate(hits, limit)

    @tool
    def list_users(self) -> list[SlackUser]:
        """List the people in the workspace with their handles, titles and emails."""
        return sorted(self._users.values(), key=lambda user: user.name)

    @tool
    def post_message(
        self,
        channel: ChannelName,
        text: str,
        thread_ts: Annotated[
            str | None, Field(description="Reply in this thread instead of the channel")
        ] = None,
    ) -> SlackMessage:
        """Post a message to a channel, or reply in a thread."""
        name = self._channel(channel)
        if not text.strip():
            raise ToolError("text is empty")
        messages = self._messages.setdefault(name, [])
        if thread_ts and not any(message.ts == thread_ts for message in messages):
            raise ToolError(f"No thread {thread_ts} in #{name}")
        latest = max(
            (m.time for ms in self._messages.values() for m in ms),
            default=datetime.combine(self.context.today, datetime.min.time(), UTC),
        )
        when = latest + timedelta(minutes=1)
        message = SlackMessage(
            channel=name,
            ts=f"{when.timestamp():.6f}",
            time=when,
            author="assistant",
            text=text,
            thread_ts=thread_ts,
        )
        messages.append(message)
        self._channels[name].message_count = len(messages)
        where = f"#{name}" + (f" thread {thread_ts}" if thread_ts else "")
        self.record(
            "post_message",
            {"channel": name, "text": text, "thread_ts": thread_ts},
            f"posted in {where}",
        )
        return message


def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)
