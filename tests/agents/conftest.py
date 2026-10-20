"""A model provider reached directly rather than through the HUD gateway."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from mcp.types import TextContent

from hud.agents.types import ToolStep
from tests.fixtures.models import Models
from tests.fixtures.services import FakeServices

if TYPE_CHECKING:
    from collections.abc import Iterator

    from hud.eval import Run


@pytest.fixture(scope="session")
def provider_server() -> Iterator[FakeServices]:
    server = FakeServices()
    server.start()
    yield server
    server.stop()


@pytest.fixture
def provider(provider_server: FakeServices) -> Models:
    """Scripted models served as a provider's own API."""
    provider_server.reset()
    return Models(provider_server)


def tool_results(run: Run) -> list[tuple[bool, str]]:
    """Each tool step on ``run``'s trace, in order, as (whether it failed, its text)."""
    results = []
    for step in run.trace.steps:
        if isinstance(step, ToolStep):
            assert step.result is not None
            text = "".join(b.text for b in step.result.content if isinstance(b, TextContent))
            results.append((step.result.isError, text))
    return results
