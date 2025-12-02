"""
Apply patch tool implementation conforming to OpenAI's apply_patch tool specification.
https://platform.openai.com/docs/guides/tools-apply-patch

Key features:
- Supports create_file, update_file, delete_file operations
- Parses V4A diff format
- Returns apply_patch_call_output format with status and output
"""

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


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


@dataclass
class ApplyPatchResult:
    """Result of apply_patch tool execution, conforming to apply_patch_call_output format."""

    status: Literal["completed", "failed"]
    output: str

    def to_dict(self) -> dict:
        return {"status": self.status, "output": self.output}


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


class ApplyPatchTool:
    """
    A tool that allows the agent to create, update, and delete files using structured diffs.
    Conforms to OpenAI's apply_patch tool specification.

    Features:
    - Supports create_file, update_file, delete_file operations
    - Parses V4A diff format
    - Returns apply_patch_call_output format
    - Path validation to prevent directory traversal
    """

    def __init__(self, base_path: str = ".") -> None:
        """
        Initialize the apply patch tool.

        Args:
            base_path: Base directory for file operations. Paths are relative to this.
        """
        self.base_path = os.path.abspath(base_path)

    def _validate_path(self, path: str) -> str:
        """Validate and resolve a path, preventing directory traversal."""
        if path.startswith("/"):
            raise DiffError(f"Absolute paths are not allowed: {path}")

        # Normalize and resolve
        full_path = os.path.normpath(os.path.join(self.base_path, path))

        # Check for directory traversal
        # Use base_path + os.sep to prevent sibling directory prefix bypass
        # e.g., /tmp/myapp_sibling shouldn't match base_path /tmp/myapp
        if full_path != self.base_path and not full_path.startswith(self.base_path + os.sep):
            raise DiffError(f"Path traversal detected: {path}")

        return full_path

    def _open_file(self, path: str) -> str:
        """Read a file's contents."""
        full_path = self._validate_path(path)
        try:
            with open(full_path) as f:
                return f.read()
        except FileNotFoundError:
            raise DiffError(f"File not found: {path}") from None
        except Exception as e:
            raise DiffError(f"Error reading file {path}: {e}") from e

    def _write_file(self, path: str, content: str) -> None:
        """Write content to a file, creating directories if needed."""
        full_path = self._validate_path(path)
        parent = os.path.dirname(full_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)

    def _remove_file(self, path: str) -> None:
        """Remove a file."""
        full_path = self._validate_path(path)
        os.remove(full_path)

    def _load_files(self, paths: list[str]) -> dict[str, str]:
        """Load multiple files into a dictionary."""
        orig = {}
        for path in paths:
            orig[path] = self._open_file(path)
        return orig

    def _process_v4a_diff(self, diff_text: str) -> str:
        """Process a V4A diff and apply it to files."""
        if not diff_text.strip().startswith("*** Begin Patch"):
            # Wrap in patch markers if not present
            diff_text = f"*** Begin Patch\n{diff_text}\n*** End Patch"

        paths = _identify_files_needed(diff_text)
        orig = self._load_files(paths)
        patch, _ = _text_to_patch(diff_text, orig)
        commit = _patch_to_commit(patch, orig)
        _apply_commit(commit, self._write_file, self._remove_file)

        changed_files = list(commit.changes.keys())
        return f"Applied patch to {len(changed_files)} file(s): {', '.join(changed_files)}"

    async def __call__(
        self,
        type: str | None = None,
        path: str | None = None,
        diff: str | None = None,
        **kwargs: object,
    ) -> ApplyPatchResult:
        """
        Apply a patch operation.

        Args:
            type: Operation type - "create_file", "update_file", or "delete_file"
            path: The file path to operate on
            diff: The V4A diff content (required for create_file and update_file)

        Returns:
            ApplyPatchResult conforming to apply_patch_call_output format.
        """
        op_type = type

        if not op_type:
            return ApplyPatchResult(
                status="failed",
                output="Error: Missing operation type",
            )

        if not path:
            return ApplyPatchResult(
                status="failed",
                output="Error: Missing file path",
            )

        try:
            if op_type == "delete_file":
                # Delete file operation
                full_path = self._validate_path(path)
                if not os.path.exists(full_path):
                    return ApplyPatchResult(
                        status="failed",
                        output=f"Error: File not found at path '{path}'",
                    )
                self._remove_file(path)
                return ApplyPatchResult(
                    status="completed",
                    output=f"Deleted {path}",
                )

            elif op_type == "create_file":
                # Create file operation
                if not diff:
                    return ApplyPatchResult(
                        status="failed",
                        output="Error: Missing diff for create_file operation",
                    )

                full_path = self._validate_path(path)
                if os.path.exists(full_path):
                    return ApplyPatchResult(
                        status="failed",
                        output=f"Error: File already exists at path '{path}'",
                    )

                # For create_file, the diff should represent the full file content
                # Parse the V4A diff format for new file
                content = self._parse_create_diff(diff)
                self._write_file(path, content)
                return ApplyPatchResult(
                    status="completed",
                    output=f"Created {path}",
                )

            elif op_type == "update_file":
                # Update file operation
                if not diff:
                    return ApplyPatchResult(
                        status="failed",
                        output="Error: Missing diff for update_file operation",
                    )

                full_path = self._validate_path(path)
                if not os.path.exists(full_path):
                    return ApplyPatchResult(
                        status="failed",
                        output=f"Error: File not found at path '{path}'",
                    )

                # Apply the V4A diff
                result = self._apply_update_diff(path, diff)
                return ApplyPatchResult(
                    status="completed",
                    output=result,
                )

            else:
                return ApplyPatchResult(
                    status="failed",
                    output=f"Error: Unknown operation type '{op_type}'",
                )

        except DiffError as e:
            return ApplyPatchResult(
                status="failed",
                output=f"Error: {e}",
            )
        except Exception as e:
            return ApplyPatchResult(
                status="failed",
                output=f"Error: {e}",
            )

    def _parse_create_diff(self, diff: str) -> str:
        """Parse a create diff and extract the file content."""
        lines = diff.strip().split("\n")
        content_lines = []

        for line in lines:
            # Skip empty lines at start
            if not line and not content_lines:
                continue
            # Lines starting with + are additions (the file content)
            if line.startswith("+"):  # noqa: SIM114
                content_lines.append(line[1:])
            elif line.startswith(" "):
                content_lines.append(line[1:])
            elif line == "":
                content_lines.append("")

        return "\n".join(content_lines)

    def _apply_update_diff(self, path: str, diff: str) -> str:
        """Apply an update diff to an existing file."""
        # Read current content
        current_content = self._open_file(path)

        # Construct full patch text
        patch_text = f"*** Begin Patch\n*** Update File: {path}\n{diff}\n*** End Patch"

        # Parse and apply
        orig = {path: current_content}
        patch, fuzz = _text_to_patch(patch_text, orig)
        commit = _patch_to_commit(patch, orig)
        _apply_commit(commit, self._write_file, self._remove_file)

        return f"Updated {path}" + (f" (fuzz: {fuzz})" if fuzz > 0 else "")
