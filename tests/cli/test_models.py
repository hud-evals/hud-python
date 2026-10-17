"""``hud models``: the gateway catalog, forks, checkpoints and the served head."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from tests.support import Reply

if TYPE_CHECKING:
    from tests.support import Hud, Platform

SONNET_ID = "00000000-0000-4000-a000-000000000001"
MINE_ID = "00000000-0000-4000-a000-000000000002"
CATALOG = [
    {
        "id": MINE_ID,
        "name": "owner/model + version",
        "model_name": "team/my-sonnet",
        "sdk_agent_type": "claude",
        "is_trainable": True,
        "provider": {"name": "anthropic"},
    },
    {
        "id": SONNET_ID,
        "name": "Claude Sonnet",
        "model_name": "claude-sonnet-4-6",
        "sdk_agent_type": "claude",
        "provider": {"name": "anthropic"},
    },
]
CHECKPOINTS = [
    {"id": "ckpt-0001-base", "name": "base", "created_at": "2026-09-01T00:00:00Z"},
    {
        "id": "ckpt-0002-step",
        "name": "step-1",
        "checkpoint_name": "sampler-1",
        "is_active": True,
        "mean_reward": 0.5,
        "loss_fn": "importance_sampling",
        "num_traces": 32,
        "created_at": "2026-09-02T00:00:00Z",
    },
]


@pytest.fixture
def catalog(platform: Platform) -> Platform:
    for model in CATALOG:
        platform.add_model(**model)
    platform.checkpoints[MINE_ID] = [dict(checkpoint) for checkpoint in CHECKPOINTS]
    return platform


def test_quiet_lists_the_model_slugs_one_per_line(hud: Hud, catalog: Platform) -> None:
    result = hud("models", "list", "--quiet")

    assert (result.exit_code, result.stdout.split("\n")[:-1]) == (
        0,
        ["claude-sonnet-4-6", "team/my-sonnet"],
    ), result


def test_models_list_reads_every_page_and_sorts_by_name(hud: Hud, catalog: Platform) -> None:
    catalog.page_size = 1

    result = hud("models", "list", "--json")

    assert result.exit_code == 0, result
    assert [model["id"] for model in result.json] == [SONNET_ID, MINE_ID]


@pytest.mark.parametrize("model", ["owner/model + version", MINE_ID])
def test_head_is_the_resolved_models_active_checkpoint(
    hud: Hud, catalog: Platform, model: str
) -> None:
    head = hud("models", "head", model, "--json")

    assert head.exit_code == 0, head
    assert (head.json["model_id"], head.json["head"]["id"]) == (MINE_ID, "ckpt-0002-step")


def test_head_set_asks_the_platform_to_serve_the_checkpoint(hud: Hud, catalog: Platform) -> None:
    result = hud("models", "head", "my-sonnet", "--set", "ckpt-0001-base", "--yes", "--json")

    assert result.exit_code == 0, result
    assert catalog.heads == {MINE_ID: "ckpt-0001-base"}


def test_a_model_with_no_checkpoints_has_no_head(hud: Hud, catalog: Platform) -> None:
    head = hud("models", "head", "claude-sonnet-4-6", "--json")

    assert (head.exit_code, head.json["head"]) == (0, None), head


def test_fork_creates_a_trainable_model_from_its_source(hud: Hud, catalog: Platform) -> None:
    result = hud("models", "fork", "Claude Sonnet", "--name", "fresh", "--yes", "--json")

    assert result.exit_code == 0, result
    (forked,) = [model for model in catalog.models if model["id"] == result.json["id"]]
    assert forked["forked_from"] == SONNET_ID


@pytest.mark.parametrize(
    ("argv", "exit_code", "outcome"),
    [
        pytest.param(["--if-not-exists"], 0, {"id": MINE_ID, "existed": True}, id="allowed"),
        pytest.param([], 1, {"error": "conflict"}, id="refused"),
    ],
)
def test_forking_onto_an_existing_name(
    hud: Hud, catalog: Platform, argv: list[str], exit_code: int, outcome: dict[str, Any]
) -> None:
    catalog.services.route(
        "api", "POST", "/v2/models/fork", Reply(status=409, json={"detail": "Model exists"})
    )

    result = hud("models", "fork", "Claude Sonnet", "--name", "my-sonnet", *argv, "--yes", "--json")

    assert result.exit_code == exit_code, result
    assert {key: result.json[key] for key in outcome} == outcome


def test_a_misspelt_model_suggests_the_close_name(hud: Hud, catalog: Platform) -> None:
    result = hud("models", "fork", "claude-sonet-4-6", "--name", "mine", "--yes", "--json")

    assert (result.exit_code, result.json["error"]) == (2, "usage"), result
    assert "claude-sonnet-4-6" in result.json["message"]
