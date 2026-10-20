"""``hud models``: the gateway catalog, forking a trainable model, and a model's
checkpoints and the head the gateway serves."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.support import Reply

if TYPE_CHECKING:
    from tests.support import FakeServices, Hud, Request

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
    """The gateway catalog, served one model a page."""

    def one_a_page(request: Request) -> Reply:
        offset = int(request.query["offset"][0])
        return Reply(json={"items": CATALOG[offset : offset + 1], "total": len(CATALOG)})

    signed_in.route("api", "GET", "/v2/models", handler=one_a_page)
    signed_in.route("api", "GET", f"/v2/models/{MODEL_ID}/checkpoints", json=CHECKPOINTS)
    signed_in.route("api", "GET", f"/v2/models/{BASE_ID}/checkpoints", json=[])
    return signed_in


def test_list_reads_the_whole_catalog_sorted_by_name(catalog: FakeServices, hud: Hud) -> None:
    listed = hud("models", "list", "--json")
    quiet = hud("models", "list", "--quiet")

    assert [model["id"] for model in listed.json] == [BASE_ID, MODEL_ID]
    assert quiet.stdout.split() == ["claude-sonnet-4-6", "team/my-sonnet"]


@pytest.mark.parametrize("source", ["Claude Sonnet", "claude-sonnet-4-6", BASE_ID])
def test_fork_derives_a_model_from_the_one_its_reference_names(
    source: str, catalog: FakeServices, hud: Hud
) -> None:
    catalog.route("api", "POST", "/v2/models/fork", json={**CATALOG[1], "id": "new"})

    result = hud("models", "fork", source, "--name", "fresh", "--yes", "--json")

    assert (result.exit_code, result.json["id"]) == (0, "new"), result
    assert catalog.bodies("api", "POST", "/v2/models/fork") == [
        {"source_model_id": BASE_ID, "name": "fresh"}
    ]


def test_fork_if_not_exists_onto_a_taken_name_returns_the_existing_model(
    catalog: FakeServices, hud: Hud
) -> None:
    catalog.route("api", "POST", "/v2/models/fork", status=409, json={"detail": "Model exists"})

    result = hud(
        "models",
        "fork",
        "claude-sonnet-4-6",
        "--name",
        "my-sonnet",
        "--if-not-exists",
        "--yes",
        "--json",
    )

    assert (result.exit_code, result.json["id"], result.json["existed"]) == (0, MODEL_ID, True)


def test_a_misspelt_model_is_refused_naming_the_close_matches(
    catalog: FakeServices, hud: Hud
) -> None:
    result = hud("models", "fork", "claude-sonet-4-6", "--name", "mine", "--yes", "--json")

    assert (result.exit_code, result.json["error"]) == (2, "usage"), result
    assert "Did you mean: claude-sonnet-4-6" in result.json["message"]
    assert catalog.requests("api", "POST") == []


def test_checkpoints_lists_the_models_checkpoints_oldest_first(
    catalog: FakeServices, hud: Hud
) -> None:
    listed = hud("models", "checkpoints", "team/my-sonnet", "--json")
    quiet = hud("models", "checkpoints", "my-sonnet", "--quiet")

    assert [c["id"] for c in listed.json] == ["ckpt-1", "ckpt-2"]
    assert quiet.stdout.split() == ["ckpt-1", "ckpt-2"]


@pytest.mark.parametrize(
    ("model", "head"),
    [("owner/model + version", "ckpt-2"), (MODEL_ID, "ckpt-2"), ("claude-sonnet-4-6", None)],
    ids=["by-display-name", "by-id", "a-model-without-checkpoints"],
)
def test_head_shows_the_active_checkpoint(
    model: str, head: str | None, catalog: FakeServices, hud: Hud
) -> None:
    result = hud("models", "head", model, "--json")

    assert result.exit_code == 0, result
    assert (result.json["head"] or {}).get("id") == head


def test_head_set_serves_the_checkpoint_it_names(catalog: FakeServices, hud: Hud) -> None:
    catalog.route("api", "PUT", f"/v2/models/{MODEL_ID}/head", json={})

    result = hud("models", "head", "my-sonnet", "--set", "ckpt-1", "--yes", "--json")

    assert (result.exit_code, result.json["checkpoint_id"]) == (0, "ckpt-1"), result
    assert catalog.bodies("api", "PUT", f"/v2/models/{MODEL_ID}/head") == [
        {"checkpoint_id": "ckpt-1"}
    ]
