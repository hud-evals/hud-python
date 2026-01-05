"""Context compaction and offloading.

This module implements the following principles:
- Compaction (reversible): Keep IDs, paths, strip content. Can reconstruct.
- Summarization (irreversible): Only when approaching rot threshold (128K tokens).
- Offloading: Store large results as files, keep only path reference.
- File System as Context: Treat filesystem as unlimited, persistent external memory.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from hud.multi_agent.schemas import ContextEntry, ContextEntryType

if TYPE_CHECKING:
    from hud.multi_agent.context import AppendOnlyContext
    from hud.multi_agent.memory import FilesystemMemory

logger = logging.getLogger(__name__)


# Default offload threshold (lowered per the following principle: "File System as Context")
# Aggressive offloading keeps context lean and uses filesystem as external memory
DEFAULT_OFFLOAD_THRESHOLD = 500  # characters (was 2000)


class ContextOffloader:
    """Offload large content to filesystem.

    When tool results or file contents exceed the threshold, they are
    stored as files and replaced with a reference in the context.

    Per the following principle: "Compression should be RESTORABLE. A URL can replace
    page content. A file path can replace document content."

    Example:
        offloader = ContextOffloader(memory, threshold=500)
        result = await offloader.process_tool_result(large_output)
        # Returns: "[Result saved to ./workspace/result_abc123.txt. Use grep to access.]"
    """

    def __init__(
        self,
        memory: FilesystemMemory,
        threshold: int = DEFAULT_OFFLOAD_THRESHOLD,
    ) -> None:
        """Initialize the offloader.

        Args:
            memory: FilesystemMemory instance for storage
            threshold: Character threshold for offloading (default: 500)
        """
        self.memory = memory
        self.threshold = threshold

    async def process_tool_result(self, result: str, tool_name: str | None = None) -> str:
        """Process a tool result, offloading if too large.

        Args:
            result: The tool result content
            tool_name: Optional tool name for the key

        Returns:
            Original result if small, or reference string if offloaded
        """
        if len(result) < self.threshold:
            return result

        # Generate key for storage
        key = f"result_{tool_name or 'tool'}_{uuid4().hex[:8]}"

        # Store and return reference
        ref = await self.memory.store(key, result)
        path = ref.replace("[Stored at: ", "").replace("]", "")

        return f"[Result saved to {path}. Use `read_file` or `grep` to access.]"

    async def process_file_content(
        self,
        content: str,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        """Process file content, offloading if too large.

        Args:
            content: File content
            path: Original file path
            start_line: Start line if partial read
            end_line: End line if partial read

        Returns:
            Original content if small, or reference string if offloaded
        """
        if len(content) < self.threshold:
            return content

        # For file content, we don't store a copy - just return a reference
        lines_info = f" (lines {start_line}-{end_line})" if start_line else ""
        return f"[File content from {path}{lines_info} is large. Use `grep` to search or `read_file` to read specific sections.]"

    def should_offload(self, content: str) -> bool:
        """Check if content should be offloaded."""
        return len(content) >= self.threshold


class ContextCompactor:
    """Compact context entries reversibly.

    Compaction keeps IDs, paths, and metadata but strips content.
    The original can be reconstructed by reading from the filesystem.

    Summarization is irreversible and only triggered near the rot threshold.
    """

    ROT_THRESHOLD = 128_000  # tokens

    def __init__(
        self,
        memory: FilesystemMemory | None = None,
        rot_threshold: int = 128_000,
    ) -> None:
        """Initialize the compactor.

        Args:
            memory: Optional FilesystemMemory for storing compacted content
            rot_threshold: Token threshold for summarization
        """
        self.memory = memory
        self.rot_threshold = rot_threshold

    def compact(self, entry: ContextEntry) -> ContextEntry:
        """Reversibly compact an entry.

        Keeps IDs, paths, and metadata. Strips content.
        Can be reconstructed by reading from filesystem.

        Args:
            entry: The entry to compact

        Returns:
            Compacted entry (or original if not compactable)
        """
        if entry.compacted:
            return entry

        if entry.type == ContextEntryType.FILE_CONTENT:
            # Convert to file reference
            return ContextEntry(
                id=entry.id,
                type=ContextEntryType.FILE_REF,
                content="",
                timestamp=entry.timestamp,
                path=entry.path,
                start_line=entry.start_line,
                end_line=entry.end_line,
                agent_id=entry.agent_id,
                token_count=10,  # Reference is small
                compacted=True,
            )

        elif entry.type == ContextEntryType.TOOL_RESULT:
            # Keep summary only
            summary = entry.content[:100].replace("\n", " ").strip()
            if len(entry.content) > 100:
                summary += "..."

            return ContextEntry(
                id=entry.id,
                type=entry.type,
                content=f"[Compacted: {summary}]",
                timestamp=entry.timestamp,
                tool_name=entry.tool_name,
                agent_id=entry.agent_id,
                token_count=len(summary) // 4,
                compacted=True,
            )

        # Other types are not compactable
        return entry

    def compact_context(self, context: AppendOnlyContext) -> int:
        """Compact all compactable entries in context.

        Only compacts entries after the frozen prefix.

        Args:
            context: The context to compact

        Returns:
            Number of entries compacted
        """
        compacted_count = 0

        for entry in context.get_compactable_entries():
            compacted = self.compact(entry)
            if compacted != entry:
                context.replace_entry(entry.id, compacted)
                compacted_count += 1

        return compacted_count

    async def summarize(
        self,
        entries: list[ContextEntry],
        llm_fn: Any = None,
    ) -> ContextEntry:
        """Irreversibly summarize entries.

        Only use when approaching rot threshold. Uses structured schema
        to reduce hallucination.

        Args:
            entries: Entries to summarize
            llm_fn: Optional LLM function for summarization

        Returns:
            Summary entry
        """
        # If no LLM provided, create a simple extractive summary
        if llm_fn is None:
            summary_parts = []
            for entry in entries[:10]:  # Take first 10 entries
                if entry.content:
                    # First sentence or first 100 chars
                    first_sentence = entry.content.split(".")[0][:100]
                    summary_parts.append(f"- {first_sentence}")

            summary_content = "Summary of previous context:\n" + "\n".join(summary_parts)
        else:
            # Use LLM for summarization
            context_text = "\n".join(e.render() for e in entries)
            summary_content = await llm_fn(
                f"Summarize the following context concisely:\n\n{context_text}"
            )

        return ContextEntry(
            id=f"summary_{uuid4().hex[:8]}",
            type=ContextEntryType.SUMMARY,
            content=summary_content,
            token_count=len(summary_content) // 4,
            compacted=True,
        )

    def should_summarize(self, context: AppendOnlyContext) -> bool:
        """Check if context needs summarization."""
        return context.token_count > self.rot_threshold

    def should_compact(self, context: AppendOnlyContext) -> bool:
        """Check if context should be compacted (80% of threshold)."""
        return context.token_count > self.rot_threshold * 0.8


__all__ = ["ContextOffloader", "ContextCompactor"]

