"""``hud models fork``: forking a model from the one a reference names."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.fixtures.services import Reply

if TYPE_CHECKING:
    from tests.conftest import Hud
    from tests.fixtures.services import FakeServices, Request

BASE_ID = "cccccccc-0000-4000-8000-000000000101"
MODEL_ID = "cccccccc-0000-4000-8000-000000000102"
CATALOG = [
    {
        "id": BASE_ID,
        "name": "Claude Sonnet",
        "model_name": "claude-sonnet-4-6",
        "sdk_agent_type": "claude",
        "provider": {"name": "anthropic"},
    },
    {
        "id": MODEL_ID,
        "name": "owner/model + version",
        "model_name": "team/my-sonnet",
        "sdk_agent_type": "claude",
        "is_trainable": True,
        "provider": {"name": "anthropic"},
    },
]
CHECKPOINTS = [
    {"id": "ckpt-1", "created_at": "2026-09-01T00:00:00Z"},
    {"id": "ckpt-2", "name": "step-1", "created_at": "2026-09-02T00:00:00Z", "is_active": True},
]


@pytest.fixture
def catalog(signed_in: FakeServices) -> FakeServices:
    """The gateway catalog, served one model a page in reverse name order."""

    def one_a_page(request: Request) -> Reply:
        offset = int(request.query["offset"][0])
        return Reply(json={"items": CATALOG[::-1][offset : offset + 1], "total": len(CATALOG)})

    signed_in.route("api", "GET", "/v2/models", handler=one_a_page)
    signed_in.route("api", "GET", f"/v2/models/{MODEL_ID}/checkpoints", json=CHECKPOINTS)
    signed_in.route("api", "GET", f"/v2/models/{BASE_ID}/checkpoints", json=[])
    return signed_in


@pytest.mark.parametrize("source", ["Claude Sonnet", "claude-sonnet-4-6"])
def test_fork_derives_a_model_from_the_one_its_reference_names(
    source: str, catalog: FakeServices, hud: Hud
) -> None:
    catalog.route("api", "POST", "/v2/models/fork", json=CATALOG[1])

    result = hud("models", "fork", source, "--name", "fresh", "--yes", "--json")

    assert result.exit_code == 0, result
    assert catalog.bodies("api", "POST", "/v2/models/fork") == [
        {"source_model_id": BASE_ID, "name": "fresh"}
    ]


def test_a_misspelt_model_is_refused(catalog: FakeServices, hud: Hud) -> None:
    result = hud("models", "fork", "claude-sonet-4-6", "--name", "mine", "--yes", "--json")

    assert (result.exit_code, result.json["error"]) == (2, "usage"), result
    assert catalog.requests("api", "POST") == []
