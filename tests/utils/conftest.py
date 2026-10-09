"""A model provider reached directly, apart from the HUD gateway."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.harness import FakeServices, Models

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(scope="session")
def provider_server() -> Iterator[FakeServices]:
    server = FakeServices()
    server.start()
    yield server
    server.stop()


@pytest.fixture
def provider(provider_server: FakeServices) -> Models:
    """Scripted models served as a provider's own API (not behind the HUD gateway)."""
    provider_server.reset()
    return Models(provider_server)
