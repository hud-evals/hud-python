"""Gemini Memory tool for persistent fact storage.

This tool matches Gemini CLI's memory tool interface:
- Simple save_memory(fact) command
- Appends facts as bullet points to a markdown file
- Facts stored under "## Gemini Added Memories" section

See: https://github.com/google-gemini/gemini-cli
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from pathlib import Path

from hud.tools.memory.base import BaseFileMemoryTool
from hud.tools.native_types import NativeToolSpec, NativeToolSpecs
from hud.tools.types import ContentResult, ToolError
from hud.types import AgentType

if TYPE_CHECKING:
    from mcp.types import ContentBlock

LOGGER = logging.getLogger(__name__)

DEFAULT_MEMORY_FILENAME = "GEMINI.md"
MEMORY_SECTION_HEADER = "## Gemini Added Memories"


class GeminiMemoryTool(BaseFileMemoryTool):
    """Persistent memory tool for Gemini agents.

    Saves facts to a markdown file (default: GEMINI.md) under a
    dedicated "## Gemini Added Memories" section.

    This tool is used when:
    - User explicitly asks to remember something
    - User states a clear, concise fact worth retaining

    Do NOT use for:
    - Conversational context only relevant to current session
    - Long, complex text (facts should be short)

    Parameters:
        fact: The specific fact to remember (required)

    Example:
        >>> tool = GeminiMemoryTool(memory_dir="./workspace")
        >>> await tool(fact="User prefers tabs over spaces")
    """

    native_specs: ClassVar[NativeToolSpecs] = {
        AgentType.GEMINI: NativeToolSpec(role="memory"),
    }

    _memory_file: Path

    def __init__(
        self,
        memory_dir: str | Path = ".",
        memory_filename: str = DEFAULT_MEMORY_FILENAME,
    ) -> None:
        """Initialize GeminiMemoryTool.

        Args:
            memory_dir: Directory for the memory file
            memory_filename: Name of the memory file (default: GEMINI.md)
        """
        super().__init__(
            base_path=memory_dir,
            memory_section_header=MEMORY_SECTION_HEADER,
        )

        self.name = "save_memory"
        self.title = "SaveMemory"
        self.description = (
            "Saves a specific piece of information or fact to your long-term memory. "
            "Use this when the user explicitly asks you to remember something, or when "
            "they state a clear, concise fact that seems important to retain for future "
            "interactions."
        )

        self._memory_file = self._base_path / memory_filename

    def _ensure_newline_separation(self, content: str) -> str:
        """Ensure proper newline separation before appending."""
        if len(content) == 0:
            return ""
        if content.endswith("\n\n"):
            return ""
        if content.endswith("\n"):
            return "\n"
        return "\n\n"

    def _compute_new_content(self, current_content: str, fact: str) -> str:
        """Compute new file content with the added memory entry.

        Args:
            current_content: Current file content
            fact: Fact to add

        Returns:
            New file content with fact appended under the memory section
        """
        # Clean up the fact (remove leading dashes)
        processed_text = fact.strip()
        processed_text = processed_text.lstrip("-").strip()
        new_memory_item = f"- {processed_text}"

        header_index = current_content.find(MEMORY_SECTION_HEADER)

        if header_index == -1:
            # Header not found - append header and entry
            separator = self._ensure_newline_separation(current_content)
            return current_content + f"{separator}{MEMORY_SECTION_HEADER}\n{new_memory_item}\n"
        else:
            # Header found - find where to insert new memory entry
            start_of_section_content = header_index + len(MEMORY_SECTION_HEADER)

            # Find next section (## ) or end of file
            next_section_index = current_content.find("\n## ", start_of_section_content)
            if next_section_index == -1:
                end_of_section_index = len(current_content)
            else:
                end_of_section_index = next_section_index

            before_section = current_content[:start_of_section_content].rstrip()
            section_content = current_content[
                start_of_section_content:end_of_section_index
            ].rstrip()
            after_section = current_content[end_of_section_index:]

            # Append new memory item
            section_content += f"\n{new_memory_item}"

            return f"{before_section}\n{section_content.lstrip()}\n{after_section}".rstrip() + "\n"

    async def __call__(
        self,
        fact: str,
    ) -> list[ContentBlock]:
        """Save a fact to memory.

        Args:
            fact: The fact or piece of information to remember

        Returns:
            List of ContentBlocks with confirmation message
        """
        if not fact or fact.strip() == "":
            raise ToolError("Parameter 'fact' must be a non-empty string.")

        # Read current content
        current_content = self.read_memory_file(self._memory_file)

        # Compute new content
        new_content = self._compute_new_content(current_content, fact)

        # Write updated content
        try:
            self.write_memory_file(self._memory_file, new_content)
        except Exception as e:
            LOGGER.error("Failed to save memory: %s", e)
            raise ToolError(f"Failed to save memory: {e}") from None

        success_message = f'Okay, I\'ve remembered that: "{fact}"'
        return ContentResult(output=success_message).to_content_blocks()

    def get_all_memories(self) -> list[str]:
        """Get all stored memories as a list.

        Returns:
            List of memory strings (without bullet points)
        """
        content = self.read_memory_file(self._memory_file)

        if MEMORY_SECTION_HEADER not in content:
            return []

        header_index = content.find(MEMORY_SECTION_HEADER)
        start = header_index + len(MEMORY_SECTION_HEADER)

        # Find next section or end
        next_section = content.find("\n## ", start)
        section = content[start:] if next_section == -1 else content[start:next_section]

        # Parse bullet points
        memories = []
        for line in section.strip().split("\n"):
            line = line.strip()
            if line.startswith("- "):
                memories.append(line[2:])

        return memories


__all__ = ["GeminiMemoryTool"]
