"""Fixture environments: each reward is 1 only when the path it names worked.

Every module is a real environment source, so any placement can serve it: in
this process, in a child process, in a container, or attached by address.
Scenarios load them by path with ``source``. Modules with a ``tasks`` list also
load the way a user's tasks file does, through ``hud eval`` or ``Taskset.from_file``.

``capable_model`` answers a fixture the way a capable model would, from what the
agent actually sent.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from tests.fixtures.models import Turn, call

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.fixtures.models import ModelRequest

HERE = Path(__file__).parent


def source(name: str) -> Path:
    """The path of fixture environment ``name``."""
    path = HERE / f"{name}.py"
    assert path.is_file(), path
    return path


def repeat_word(request: ModelRequest) -> Turn:
    """Answer ``echo`` and ``verifier``: the word the prompt asks for."""
    match = re.search(r"Reply with: (\S+)", request.prompt)
    return Turn(text=match.group(1) if match else "no word in the prompt")


def collect_digits() -> Callable[[ModelRequest], Turn]:
    """Answer ``tools``: call next_digit until it says done, then send every digit.

    Responses chains requests and sends only the newest tool result, so the policy
    keeps the digits it has seen, as the model's own context would.
    """
    digits: list[str] = []

    def policy(request: ModelRequest) -> Turn:
        if request.tool_results:
            latest = re.search(r"\b(\d|done)\b", request.tool_results[-1])
            assert latest is not None, request.tool_results[-1]
            if latest.group(1) == "done":
                return Turn(text="".join(digits))
            digits.append(latest.group(1))
        (tool,) = [name for name in request.tools if name.endswith("next_digit")]
        return call(tool)

    return policy


def read_token(request: ModelRequest) -> Turn:
    """Answer ``secret``: cat the file the prompt names, then send the token it printed."""
    if not request.tool_results:
        path = re.search(r"Read the file (\S+) and", request.prompt)
        assert path is not None, request.prompt
        return Turn(shell=f"cat {path.group(1)}")
    token = re.search(r"(?<![0-9a-f])[0-9a-f]{16}(?![0-9a-f])", request.tool_results[-1])
    return Turn(text=token.group(0) if token else "no token in the tool result")


def capable_model(fixture: str) -> Callable[[ModelRequest], Turn]:
    """A scripted model that takes fixture environment ``fixture`` to its reward."""
    match fixture:
        case "echo" | "verifier":
            return repeat_word
        case "secret":
            return read_token
        case "tools":
            return collect_digits()
    raise ValueError(f"no scripted model answers {fixture!r}")
