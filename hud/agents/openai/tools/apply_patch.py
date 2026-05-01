"""OpenAI apply_patch parser helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


class DiffError(ValueError):
    """Exception raised when diff parsing or application fails."""


class ActionType(str, Enum):
    ADD = "add"
    DELETE = "delete"
    UPDATE = "update"


@dataclass
class FileChange:
    type: ActionType
    old_content: str | None = None
    new_content: str | None = None
    move_path: str | None = None


@dataclass
class Commit:
    changes: dict[str, FileChange] = field(default_factory=dict)


@dataclass
class Chunk:
    orig_index: int = -1  # line index of the first line in the original file
    del_lines: list[str] = field(default_factory=list)
    ins_lines: list[str] = field(default_factory=list)


@dataclass
class PatchAction:
    type: ActionType
    new_file: str | None = None
    chunks: list[Chunk] = field(default_factory=list)
    move_path: str | None = None


@dataclass
class Patch:
    actions: dict[str, PatchAction] = field(default_factory=dict)


class Parser:
    """Parser for V4A diff format."""

    def __init__(self, current_files: dict[str, str], lines: list[str], index: int = 0) -> None:
        self.current_files = current_files
        self.lines = lines
        self.index = index
        self.patch = Patch()
        self.fuzz = 0

    def is_done(self, prefixes: tuple[str, ...] | None = None) -> bool:
        if self.index >= len(self.lines):
            return True
        return prefixes is not None and self.lines[self.index].startswith(prefixes)

    def startswith(self, prefix: str | tuple[str, ...]) -> bool:
        if self.index >= len(self.lines):
            raise DiffError(f"Unexpected end of patch at index {self.index}")
        return self.lines[self.index].startswith(prefix)

    def read_str(self, prefix: str = "", return_everything: bool = False) -> str:
        if self.index >= len(self.lines):
            return ""  # At EOF, no match possible
        if self.lines[self.index].startswith(prefix):
            if return_everything:
                text = self.lines[self.index]
            else:
                text = self.lines[self.index][len(prefix) :]
            self.index += 1
            return text
        return ""

    def parse(self) -> None:
        while not self.is_done(("*** End Patch",)):
            path = self.read_str("*** Update File: ")
            if path:
                if path in self.patch.actions:
                    raise DiffError(f"Update File Error: Duplicate Path: {path}")
                move_to = self.read_str("*** Move to: ")
                if path not in self.current_files:
                    raise DiffError(f"Update File Error: Missing File: {path}")
                text = self.current_files[path]
                action = self.parse_update_file(text)
                action.move_path = move_to if move_to else None
                self.patch.actions[path] = action
                continue

            path = self.read_str("*** Delete File: ")
            if path:
                if path in self.patch.actions:
                    raise DiffError(f"Delete File Error: Duplicate Path: {path}")
                if path not in self.current_files:
                    raise DiffError(f"Delete File Error: Missing File: {path}")
                self.patch.actions[path] = PatchAction(type=ActionType.DELETE)
                continue

            path = self.read_str("*** Add File: ")
            if path:
                if path in self.patch.actions:
                    raise DiffError(f"Add File Error: Duplicate Path: {path}")
                self.patch.actions[path] = self.parse_add_file()
                continue

            raise DiffError(f"Unknown Line: {self.lines[self.index]}")

        if self.index >= len(self.lines) or not self.startswith("*** End Patch"):
            raise DiffError("Missing End Patch")
        self.index += 1

    def parse_update_file(self, text: str) -> PatchAction:
        action = PatchAction(type=ActionType.UPDATE)
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
            def_str = self.read_str("@@ ")
            section_str = ""
            if not def_str and self.lines[self.index] == "@@":
                section_str = self.lines[self.index]
                self.index += 1

            if not (def_str or section_str or index == 0):
                raise DiffError(f"Invalid Line:\n{self.lines[self.index]}")

            if def_str.strip():
                found = False
                if not [s for s in lines[:index] if s == def_str]:
                    for i, s in enumerate(lines[index:], index):
                        if s == def_str:
                            index = i + 1
                            found = True
                            break

                if not found and not [s for s in lines[:index] if s.strip() == def_str.strip()]:
                    for i, s in enumerate(lines[index:], index):
                        if s.strip() == def_str.strip():
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

            for ch in chunks:
                ch.orig_index += new_index
                action.chunks.append(ch)

            index = new_index + len(next_chunk_context)
            self.index = end_patch_index

        return action

    def parse_add_file(self) -> PatchAction:
        lines = []
        while not self.is_done(
            ("*** End Patch", "*** Update File:", "*** Delete File:", "*** Add File:")
        ):
            s = self.read_str()
            if not s.startswith("+"):
                raise DiffError(f"Invalid Add File Line: {s}")
            s = s[1:]
            lines.append(s)
        return PatchAction(type=ActionType.ADD, new_file="\n".join(lines))

    def _peek_next_section(self) -> tuple[list[str], list[Chunk], int, bool]:
        old: list[str] = []
        del_lines: list[str] = []
        ins_lines: list[str] = []
        chunks: list[Chunk] = []
        mode = "keep"
        orig_index = self.index
        index = self.index

        while index < len(self.lines):
            s = self.lines[index]
            if s.startswith(
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
            if s == "***":
                break
            elif s.startswith("***"):
                raise DiffError(f"Invalid Line: {s}")

            index += 1
            last_mode = mode

            if s == "":
                s = " "

            if s[0] == "+":
                mode = "add"
            elif s[0] == "-":
                mode = "delete"
            elif s[0] == " ":
                mode = "keep"
            else:
                raise DiffError(f"Invalid Line: {s}")

            s = s[1:]

            if mode == "keep" and last_mode != mode:
                if ins_lines or del_lines:
                    chunks.append(
                        Chunk(
                            orig_index=len(old) - len(del_lines),
                            del_lines=del_lines,
                            ins_lines=ins_lines,
                        )
                    )
                del_lines = []
                ins_lines = []

            if mode == "delete":
                del_lines.append(s)
                old.append(s)
            elif mode == "add":
                ins_lines.append(s)
            elif mode == "keep":
                old.append(s)

        if ins_lines or del_lines:
            chunks.append(
                Chunk(
                    orig_index=len(old) - len(del_lines),
                    del_lines=del_lines,
                    ins_lines=ins_lines,
                )
            )

        if index < len(self.lines) and self.lines[index] == "*** End of File":
            index += 1
            return old, chunks, index, True

        if index == orig_index:
            raise DiffError(f"Nothing in this section - {index=} {self.lines[index]}")

        return old, chunks, index, False


def _find_context_core(lines: list[str], context: list[str], start: int) -> tuple[int, int]:
    if not context:
        return start, 0

    # Prefer identical
    for i in range(start, len(lines)):
        if lines[i : i + len(context)] == context:
            return i, 0

    # RStrip is ok
    for i in range(start, len(lines)):
        if [s.rstrip() for s in lines[i : i + len(context)]] == [s.rstrip() for s in context]:
            return i, 1

    # Fine, Strip is ok too
    for i in range(start, len(lines)):
        if [s.strip() for s in lines[i : i + len(context)]] == [s.strip() for s in context]:
            return i, 100

    return -1, 0


def _find_context(lines: list[str], context: list[str], start: int, eof: bool) -> tuple[int, int]:
    if eof:
        new_index, fuzz = _find_context_core(lines, context, len(lines) - len(context))
        if new_index != -1:
            return new_index, fuzz
        new_index, fuzz = _find_context_core(lines, context, start)
        return new_index, fuzz + 10000
    return _find_context_core(lines, context, start)


def _get_updated_file(text: str, action: PatchAction, path: str) -> str:
    assert action.type == ActionType.UPDATE
    orig_lines = text.split("\n")
    dest_lines = []
    orig_index = 0

    for chunk in action.chunks:
        if chunk.orig_index > len(orig_lines):
            raise DiffError(
                f"_get_updated_file: {path}: chunk.orig_index {chunk.orig_index} "
                f"> len(lines) {len(orig_lines)}"
            )
        if orig_index > chunk.orig_index:
            raise DiffError(
                f"_get_updated_file: {path}: orig_index {orig_index} "
                f"> chunk.orig_index {chunk.orig_index}"
            )

        dest_lines.extend(orig_lines[orig_index : chunk.orig_index])
        orig_index = chunk.orig_index

        if chunk.ins_lines:
            dest_lines.extend(chunk.ins_lines)

        orig_index += len(chunk.del_lines)

    dest_lines.extend(orig_lines[orig_index:])
    return "\n".join(dest_lines)


def _text_to_patch(text: str, orig: dict[str, str]) -> tuple[Patch, int]:
    lines = text.strip().split("\n")
    if len(lines) < 2 or not lines[0].startswith("*** Begin Patch") or lines[-1] != "*** End Patch":
        raise DiffError("Invalid patch text")

    parser = Parser(current_files=orig, lines=lines, index=1)
    parser.parse()
    return parser.patch, parser.fuzz


def _identify_files_needed(text: str) -> list[str]:
    lines = text.strip().split("\n")
    result = set()
    for line in lines:
        if line.startswith("*** Update File: "):
            result.add(line[len("*** Update File: ") :])
        if line.startswith("*** Delete File: "):
            result.add(line[len("*** Delete File: ") :])
    return list(result)


def _patch_to_commit(patch: Patch, orig: dict[str, str]) -> Commit:
    commit = Commit()
    for path, action in patch.actions.items():
        if action.type == ActionType.DELETE:
            commit.changes[path] = FileChange(type=ActionType.DELETE, old_content=orig[path])
        elif action.type == ActionType.ADD:
            commit.changes[path] = FileChange(type=ActionType.ADD, new_content=action.new_file)
        elif action.type == ActionType.UPDATE:
            new_content = _get_updated_file(text=orig[path], action=action, path=path)
            commit.changes[path] = FileChange(
                type=ActionType.UPDATE,
                old_content=orig[path],
                new_content=new_content,
                move_path=action.move_path,
            )
    return commit


def _apply_commit(commit: Commit, write_fn: Callable, remove_fn: Callable) -> None:
    for path, change in commit.changes.items():
        if change.type == ActionType.DELETE:
            remove_fn(path)
        elif change.type == ActionType.ADD:
            write_fn(path, change.new_content)
        elif change.type == ActionType.UPDATE:
            if change.move_path:
                write_fn(change.move_path, change.new_content)
                remove_fn(path)
            else:
                write_fn(path, change.new_content)
