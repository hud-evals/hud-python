"""Append-only context with KV cache optimization.

This module implements the context management system following the following principles:
- Append-only: Never modify existing entries (optimizes KV cache)
- Prefix freezing: Mark stable prefix for consistent cache hits
- Token tracking: Monitor context size for compaction decisions
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from hud.multi_agent.schemas import ContextEntry, ContextEntryType


class AppendOnlyContext:
    """Context that only appends, never modifies - optimizes KV cache.

    Key principles:
    1. Entries are immutable once added
    2. Prefix can be frozen for KV cache optimization
    3. Token count is tracked for compaction decisions

    Example:
        context = AppendOnlyContext()
        context.append_system("You are a helpful assistant.")
        context.freeze_prefix()  # System prompt is now stable

        context.append_user("Hello!")
        context.append_assistant("Hi there!")
    """

    def __init__(self, max_tokens: int = 128_000) -> None:
        """Initialize the context.

        Args:
            max_tokens: Maximum tokens before triggering summarization (rot threshold)
        """
        self._entries: list[ContextEntry] = []
        self._frozen_prefix_length: int = 0
        self._max_tokens = max_tokens
        self._token_count: int = 0

    @property
    def entries(self) -> list[ContextEntry]:
        """Get all context entries (read-only view)."""
        return list(self._entries)

    @property
    def token_count(self) -> int:
        """Get estimated token count."""
        return self._token_count

    @property
    def frozen_prefix_length(self) -> int:
        """Get the length of the frozen prefix."""
        return self._frozen_prefix_length

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Simple heuristic: ~4 characters per token
        return len(text) // 4

    def _generate_id(self) -> str:
        """Generate a unique entry ID."""
        return uuid.uuid4().hex[:12]

    def append(self, entry: ContextEntry) -> None:
        """Append an entry to the context.

        Args:
            entry: The context entry to append

        Note:
            Entries before frozen_prefix_length should never be modified.
        """
        if entry.token_count is None:
            entry.token_count = self._estimate_tokens(entry.content)

        self._entries.append(entry)
        self._token_count += entry.token_count or 0

    def append_system(self, content: str, **kwargs: Any) -> ContextEntry:
        """Append a system message."""
        entry = ContextEntry(
            id=self._generate_id(),
            type=ContextEntryType.SYSTEM,
            content=content,
            **kwargs,
        )
        self.append(entry)
        return entry

    def append_user(self, content: str, **kwargs: Any) -> ContextEntry:
        """Append a user message."""
        entry = ContextEntry(
            id=self._generate_id(),
            type=ContextEntryType.USER,
            content=content,
            **kwargs,
        )
        self.append(entry)
        return entry

    def append_assistant(self, content: str, **kwargs: Any) -> ContextEntry:
        """Append an assistant message."""
        entry = ContextEntry(
            id=self._generate_id(),
            type=ContextEntryType.ASSISTANT,
            content=content,
            **kwargs,
        )
        self.append(entry)
        return entry

    def append_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        agent_id: str | None = None,
    ) -> ContextEntry:
        """Append a tool call entry."""
        content = f"Calling {tool_name} with {tool_args}"
        entry = ContextEntry(
            id=self._generate_id(),
            type=ContextEntryType.TOOL_CALL,
            content=content,
            tool_name=tool_name,
            tool_args=tool_args,
            agent_id=agent_id,
        )
        self.append(entry)
        return entry

    def append_tool_result(
        self,
        content: str,
        tool_name: str | None = None,
        agent_id: str | None = None,
    ) -> ContextEntry:
        """Append a tool result entry."""
        entry = ContextEntry(
            id=self._generate_id(),
            type=ContextEntryType.TOOL_RESULT,
            content=content,
            tool_name=tool_name,
            agent_id=agent_id,
        )
        self.append(entry)
        return entry

    def append_file_content(
        self,
        content: str,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> ContextEntry:
        """Append file content."""
        entry = ContextEntry(
            id=self._generate_id(),
            type=ContextEntryType.FILE_CONTENT,
            content=content,
            path=path,
            start_line=start_line,
            end_line=end_line,
        )
        self.append(entry)
        return entry

    def append_file_ref(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> ContextEntry:
        """Append a compacted file reference (no content)."""
        entry = ContextEntry(
            id=self._generate_id(),
            type=ContextEntryType.FILE_REF,
            content="",  # No content in reference
            path=path,
            start_line=start_line,
            end_line=end_line,
            compacted=True,
        )
        self.append(entry)
        return entry

    def append_error(self, error: str, agent_id: str | None = None) -> ContextEntry:
        """Append an error entry."""
        entry = ContextEntry(
            id=self._generate_id(),
            type=ContextEntryType.ERROR,
            content=error,
            agent_id=agent_id,
        )
        self.append(entry)
        return entry

    def freeze_prefix(self) -> None:
        """Mark current entries as stable prefix for KV cache.

        After freezing, the prefix should never be modified to maximize
        cache hits across inference calls.
        """
        self._frozen_prefix_length = len(self._entries)

    def get_frozen_prefix(self) -> list[ContextEntry]:
        """Get entries in the frozen prefix."""
        return self._entries[: self._frozen_prefix_length]

    def get_unfrozen_entries(self) -> list[ContextEntry]:
        """Get entries after the frozen prefix."""
        return self._entries[self._frozen_prefix_length :]

    def render(self) -> str:
        """Render context as a single string.

        The prefix is always identical for cache hits.
        """
        return "\n".join(e.render() for e in self._entries)

    def render_for_agent(self) -> list[dict[str, Any]]:
        """Render context as a list of messages for LLM API.

        Returns format compatible with OpenAI/Claude message format.
        """
        messages = []
        for entry in self._entries:
            if entry.type == ContextEntryType.SYSTEM:
                messages.append({"role": "system", "content": entry.content})
            elif entry.type == ContextEntryType.USER:
                messages.append({"role": "user", "content": entry.content})
            elif entry.type == ContextEntryType.ASSISTANT:
                messages.append({"role": "assistant", "content": entry.content})
            elif entry.type in (ContextEntryType.TOOL_RESULT, ContextEntryType.FILE_CONTENT):
                # Tool results go as user messages (or tool role depending on API)
                messages.append({"role": "user", "content": entry.render()})
            elif entry.type == ContextEntryType.ERROR:
                messages.append({"role": "user", "content": f"Error: {entry.content}"})
            # Skip FILE_REF and TOOL_CALL as they're metadata
        return messages

    def should_compact(self) -> bool:
        """Check if context is approaching the rot threshold."""
        return self._token_count > self._max_tokens * 0.8

    def should_summarize(self) -> bool:
        """Check if context has exceeded the rot threshold."""
        return self._token_count > self._max_tokens

    def get_compactable_entries(self) -> list[ContextEntry]:
        """Get entries that can be compacted (unfrozen, not already compacted)."""
        return [
            e
            for e in self.get_unfrozen_entries()
            if not e.compacted and e.type in (ContextEntryType.FILE_CONTENT, ContextEntryType.TOOL_RESULT)
        ]

    def replace_entry(self, entry_id: str, new_entry: ContextEntry) -> bool:
        """Replace an entry with a compacted version.

        Only allowed for entries after the frozen prefix.

        Args:
            entry_id: ID of the entry to replace
            new_entry: The replacement entry (should be compacted version)

        Returns:
            True if replacement was successful
        """
        for i, entry in enumerate(self._entries):
            if entry.id == entry_id:
                if i < self._frozen_prefix_length:
                    return False  # Cannot modify frozen prefix

                # Update token count
                old_tokens = entry.token_count or 0
                new_tokens = new_entry.token_count or self._estimate_tokens(new_entry.content)
                new_entry.token_count = new_tokens

                self._entries[i] = new_entry
                self._token_count = self._token_count - old_tokens + new_tokens
                return True
        return False

    def snapshot(self) -> dict[str, Any]:
        """Create a snapshot for checkpointing."""
        return {
            "entries": [e.model_dump() for e in self._entries],
            "frozen_prefix_length": self._frozen_prefix_length,
            "token_count": self._token_count,
            "max_tokens": self._max_tokens,
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any]) -> AppendOnlyContext:
        """Restore context from a snapshot."""
        context = cls(max_tokens=snapshot.get("max_tokens", 128_000))
        context._entries = [ContextEntry.model_validate(e) for e in snapshot["entries"]]
        context._frozen_prefix_length = snapshot["frozen_prefix_length"]
        context._token_count = snapshot["token_count"]
        return context

    def __len__(self) -> int:
        """Get number of entries."""
        return len(self._entries)

    def __repr__(self) -> str:
        return (
            f"AppendOnlyContext(entries={len(self._entries)}, "
            f"frozen={self._frozen_prefix_length}, "
            f"tokens={self._token_count}/{self._max_tokens})"
        )


__all__ = ["AppendOnlyContext"]

