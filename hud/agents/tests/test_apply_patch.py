# pyright: reportPrivateUsage=false
"""The OpenAI V4A apply-patch engine: parse a patch + apply it via callbacks.

``_text_to_patch`` parses the V4A diff text against the current files; ``_apply_patch``
applies the parsed actions through ``write``/``remove`` callbacks. Both are pure, so
the file store is just a dict captured by the callbacks.
"""

from __future__ import annotations

from typing import Any

import pytest

from hud.agents.openai.tools.apply_patch import DiffError, _apply_patch, _text_to_patch


def _apply(patch_text: str, orig: dict[str, str]) -> tuple[dict[str, str | None], list[str]]:
    actions, _fuzz = _text_to_patch(patch_text, orig)
    writes: dict[str, str | None] = {}
    removed: list[str] = []
    _apply_patch(actions, orig, lambda p, c: writes.__setitem__(p, c), removed.append)
    return writes, removed


def test_add_file_writes_new_content() -> None:
    patch = "*** Begin Patch\n*** Add File: hello.txt\n+hello\n+world\n*** End Patch"
    writes, removed = _apply(patch, {})
    assert writes == {"hello.txt": "hello\nworld"}
    assert removed == []


def test_update_file_replaces_a_line_in_context() -> None:
    orig = {"f.txt": "line1\nline2\nline3"}
    patch = (
        "*** Begin Patch\n*** Update File: f.txt\n@@\n line1\n-line2\n+LINE2\n line3\n*** End Patch"
    )
    writes, _ = _apply(patch, orig)
    assert writes["f.txt"] == "line1\nLINE2\nline3"


def test_delete_file_calls_remove() -> None:
    writes, removed = _apply(
        "*** Begin Patch\n*** Delete File: f.txt\n*** End Patch", {"f.txt": "gone"}
    )
    assert removed == ["f.txt"]
    assert writes == {}


def test_update_with_move_renames_file() -> None:
    orig = {"a.txt": "hi"}
    patch = (
        "*** Begin Patch\n*** Update File: a.txt\n*** Move to: b.txt\n@@\n-hi\n+bye\n*** End Patch"
    )
    writes, removed = _apply(patch, orig)
    assert writes == {"b.txt": "bye"}
    assert removed == ["a.txt"]


def test_invalid_patch_without_sentinels_raises() -> None:
    with pytest.raises(DiffError):
        _text_to_patch("just some text", {})


def test_update_missing_file_raises() -> None:
    patch = "*** Begin Patch\n*** Update File: ghost.txt\n@@\n-x\n+y\n*** End Patch"
    with pytest.raises(DiffError, match="Missing File"):
        _text_to_patch(patch, {})


def test_duplicate_update_path_raises() -> None:
    orig: dict[str, Any] = {"f.txt": "a"}
    patch = (
        "*** Begin Patch\n*** Update File: f.txt\n@@\n-a\n+b\n"
        "*** Update File: f.txt\n@@\n-b\n+c\n*** End Patch"
    )
    with pytest.raises(DiffError, match="Duplicate Path"):
        _text_to_patch(patch, orig)
