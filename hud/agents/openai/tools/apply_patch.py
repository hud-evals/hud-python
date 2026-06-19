# pyright: reportUnusedFunction=false
"""OpenAI apply_patch parser helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable


class DiffError(ValueError):
    """Exception raised when diff parsing or application fails."""


ActionType = Literal["add", "delete", "update"]


@dataclass
class Chunk:
    orig_index: int = -1  # line index of the first line in the original file
    del_lines: list[str] = field(default_factory=list[str])
    ins_lines: list[str] = field(default_factory=list[str])


@dataclass
class PatchAction:
    type: ActionType
    new_file: str | None = None
    chunks: list[Chunk] = field(default_factory=list[Chunk])
    move_path: str | None = None


class Parser:
    """Parser for V4A diff format."""

    def __init__(self, current_files: dict[str, str], lines: list[str], index: int = 0) -> None:
        self.current_files = current_files
        self.lines = lines
        self.index = index
        self.actions: dict[str, PatchAction] = {}
        self.fuzz = 0

    def is_done(self, prefixes: tuple[str, ...] | None = None) -> bool:
        if self.index >= len(self.lines):
            return True
        return prefixes is not None and self.lines[self.index].startswith(prefixes)

    def read_str(self, prefix: str = "") -> str:
        if self.index >= len(self.lines):
            return ""  # At EOF, no match possible
        if self.lines[self.index].startswith(prefix):
            text = self.lines[self.index][len(prefix) :]
            self.index += 1
            return text
        return ""

    def parse(self) -> None:
        while not self.is_done(("*** End Patch",)):
            path = self.read_str("*** Update File: ")
            if path:
                if path in self.actions:
                    raise DiffError(f"Update File Error: Duplicate Path: {path}")
                move_to = self.read_str("*** Move to: ")
                if path not in self.current_files:
                    raise DiffError(f"Update File Error: Missing File: {path}")
                text = self.current_files[path]
                action = self.parse_update_file(text)
                action.move_path = move_to if move_to else None
                self.actions[path] = action
                continue

            path = self.read_str("*** Delete File: ")
            if path:
                if path in self.actions:
                    raise DiffError(f"Delete File Error: Duplicate Path: {path}")
                if path not in self.current_files:
                    raise DiffError(f"Delete File Error: Missing File: {path}")
                self.actions[path] = PatchAction(type="delete")
                continue

            path = self.read_str("*** Add File: ")
            if path:
                if path in self.actions:
                    raise DiffError(f"Add File Error: Duplicate Path: {path}")
                self.actions[path] = self.parse_add_file()
                continue

            raise DiffError(f"Unknown Line: {self.lines[self.index]}")

        if self.index >= len(self.lines) or not self.lines[self.index].startswith("*** End Patch"):
            raise DiffError("Missing End Patch")
        self.index += 1

    def parse_update_file(self, text: str) -> PatchAction:
        action = PatchAction(type="update")
        lines = text.split("\n")
        index = 0

        while not self.is_done(
            (
                "*** End Patch",
                "*** Update File:",
                "*** Delete File:",
                "*** Add File:",
                "*** End of File",
            )
        ):
            section_anchor = self.read_str("@@ ")
            has_section_marker = False
            if not section_anchor and self.lines[self.index] == "@@":
                has_section_marker = True
                self.index += 1

            if not (section_anchor or has_section_marker or index == 0):
                raise DiffError(f"Invalid Line:\n{self.lines[self.index]}")

            if section_anchor.strip():
                found = False
                if not any(line == section_anchor for line in lines[:index]):
                    for i, line in enumerate(lines[index:], index):
                        if line == section_anchor:
                            index = i + 1
                            found = True
                            break

                stripped_anchor = section_anchor.strip()
                if not found and not any(line.strip() == stripped_anchor for line in lines[:index]):
                    for i, line in enumerate(lines[index:], index):
                        if line.strip() == stripped_anchor:
                            index = i + 1
                            self.fuzz += 1
                            found = True
                            break

            next_chunk_context, chunks, end_patch_index, eof = self._peek_next_section()
            next_chunk_text = "\n".join(next_chunk_context)
            new_index, fuzz = _find_context(lines, next_chunk_context, index, eof)

            if new_index == -1:
                if eof:
                    raise DiffError(f"Invalid EOF Context {index}:\n{next_chunk_text}")
                else:
                    raise DiffError(f"Invalid Context {index}:\n{next_chunk_text}")

            self.fuzz += fuzz

            for chunk in chunks:
                chunk.orig_index += new_index
                action.chunks.append(chunk)

            index = new_index + len(next_chunk_context)
            self.index = end_patch_index

        return action

    def parse_add_file(self) -> PatchAction:
        lines: list[str] = []
        while not self.is_done(
            ("*** End Patch", "*** Update File:", "*** Delete File:", "*** Add File:")
        ):
            line = self.read_str()
            if not line.startswith("+"):
                raise DiffError(f"Invalid Add File Line: {line}")
            lines.append(line[1:])
        return PatchAction(type="add", new_file="\n".join(lines))

    def _peek_next_section(self) -> tuple[list[str], list[Chunk], int, bool]:
        old: list[str] = []
        del_lines: list[str] = []
        ins_lines: list[str] = []
        chunks: list[Chunk] = []
        mode = "keep"
        orig_index = self.index
        index = self.index

        def flush_chunk() -> None:
            nonlocal del_lines, ins_lines
            if not (ins_lines or del_lines):
                return
            chunks.append(
                Chunk(
                    orig_index=len(old) - len(del_lines),
                    del_lines=del_lines,
                    ins_lines=ins_lines,
                )
            )
            del_lines = []
            ins_lines = []

        while index < len(self.lines):
            line = self.lines[index]
            if line.startswith(
                (
                    "@@",
                    "*** End Patch",
                    "*** Update File:",
                    "*** Delete File:",
                    "*** Add File:",
                    "*** End of File",
                )
            ):
                break
            if line == "***":
                break
            elif line.startswith("***"):
                raise DiffError(f"Invalid Line: {line}")

            index += 1
            last_mode = mode

            if line == "":
                line = " "

            if line[0] == "+":
                mode = "add"
            elif line[0] == "-":
                mode = "delete"
            elif line[0] == " ":
                mode = "keep"
            else:
                raise DiffError(f"Invalid Line: {line}")

            line = line[1:]

            if mode == "keep" and last_mode != mode:
                flush_chunk()

            if mode == "delete":
                del_lines.append(line)
                old.append(line)
            elif mode == "add":
                ins_lines.append(line)
            elif mode == "keep":
                old.append(line)

        flush_chunk()

        if index < len(self.lines) and self.lines[index] == "*** End of File":
            index += 1
            return old, chunks, index, True

        if index == orig_index:
            raise DiffError(f"Nothing in this section - {index=} {self.lines[index]}")

        return old, chunks, index, False


def _find_context(lines: list[str], context: list[str], start: int, eof: bool) -> tuple[int, int]:
    if not context:
        return start, 0

    search_starts = [len(lines) - len(context), start] if eof else [start]
    rstripped_context = [line.rstrip() for line in context]
    stripped_context = [line.strip() for line in context]

    for attempt, search_start in enumerate(search_starts):
        fuzz_offset = 10000 if eof and attempt > 0 else 0

        for i in range(search_start, len(lines)):
            candidate = lines[i : i + len(context)]
            if candidate == context:
                return i, fuzz_offset

        for i in range(search_start, len(lines)):
            candidate = lines[i : i + len(context)]
            if [line.rstrip() for line in candidate] == rstripped_context:
                return i, fuzz_offset + 1

        for i in range(search_start, len(lines)):
            candidate = lines[i : i + len(context)]
            if [line.strip() for line in candidate] == stripped_context:
                return i, fuzz_offset + 100

    return -1, 0


def _text_to_patch(text: str, orig: dict[str, str]) -> tuple[dict[str, PatchAction], int]:
    lines = text.strip().split("\n")
    if len(lines) < 2 or not lines[0].startswith("*** Begin Patch") or lines[-1] != "*** End Patch":
        raise DiffError("Invalid patch text")

    parser = Parser(current_files=orig, lines=lines, index=1)
    parser.parse()
    return parser.actions, parser.fuzz


def _apply_patch(
    patch: dict[str, PatchAction],
    orig: dict[str, str],
    write_fn: Callable[[str, str | None], None],
    remove_fn: Callable[[str], None],
) -> None:
    for path, action in patch.items():
        match action.type:
            case "delete":
                remove_fn(path)
            case "add":
                write_fn(path, action.new_file)
            case "update":
                orig_lines = orig[path].split("\n")
                dest_lines: list[str] = []
                orig_index = 0

                for chunk in action.chunks:
                    if chunk.orig_index > len(orig_lines):
                        raise DiffError(
                            f"_apply_patch: {path}: chunk.orig_index {chunk.orig_index} "
                            f"> len(lines) {len(orig_lines)}"
                        )
                    if orig_index > chunk.orig_index:
                        raise DiffError(
                            f"_apply_patch: {path}: orig_index {orig_index} "
                            f"> chunk.orig_index {chunk.orig_index}"
                        )

                    dest_lines.extend(orig_lines[orig_index : chunk.orig_index])
                    dest_lines.extend(chunk.ins_lines)
                    orig_index = chunk.orig_index + len(chunk.del_lines)

                dest_lines.extend(orig_lines[orig_index:])
                new_content = "\n".join(dest_lines)
                if action.move_path:
                    write_fn(action.move_path, new_content)
                    remove_fn(path)
                else:
                    write_fn(path, new_content)
