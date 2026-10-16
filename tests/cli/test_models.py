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


@pytest.mark.parametrize(
    ("argv", "stdout"),
    [
        (["models", "list", "--quiet"], ["claude-sonnet-4-6", "team/my-sonnet"]),
        (["models", "checkpoints", "my-sonnet", "--quiet"], ["ckpt-0001-base", "ckpt-0002-step"]),
    ],
)
def test_quiet_prints_one_identifier_per_line(
    hud: Hud, catalog: Platform, argv: list[str], stdout: list[str]
) -> None:
    result = hud(*argv)

    assert (result.exit_code, result.stdout.split("\n")[:-1]) == (0, stdout), result


def test_models_list_reads_every_page_and_sorts_by_name(hud: Hud, catalog: Platform) -> None:
    catalog.page_size = 1

    result = hud("models", "list", "--json")

    assert result.exit_code == 0, result
    assert [model["id"] for model in result.json] == [SONNET_ID, MINE_ID]


@pytest.mark.parametrize("model", ["owner/model + version", MINE_ID])
def test_checkpoints_and_head_read_the_resolved_model(
    hud: Hud, catalog: Platform, model: str
) -> None:
    checkpoints = hud("models", "checkpoints", model, "--json")
    head = hud("models", "head", model, "--json")

    assert (checkpoints.exit_code, head.exit_code) == (0, 0), head
    assert [checkpoint["id"] for checkpoint in checkpoints.json] == [
        "ckpt-0001-base",
        "ckpt-0002-step",
    ]
    assert (head.json["model_id"], head.json["head"]["id"]) == (MINE_ID, "ckpt-0002-step")


@pytest.mark.parametrize(("argv", "sent"), [([], {MINE_ID: "ckpt-0001-base"}), (["--dry-run"], {})])
def test_head_set_asks_the_platform_to_serve_the_checkpoint(
    hud: Hud, catalog: Platform, argv: list[str], sent: dict[str, str]
) -> None:
    result = hud("models", "head", "my-sonnet", "--set", "ckpt-0001-base", *argv, "--json")

    assert (result.exit_code, result.json["checkpoint_id"]) == (0, "ckpt-0001-base"), result
    assert catalog.heads == sent


def test_a_model_with_no_checkpoints_has_no_head(hud: Hud, catalog: Platform) -> None:
    head = hud("models", "head", "claude-sonnet-4-6", "--json")
    checkpoints = hud("models", "checkpoints", "claude-sonnet-4-6", "--json")

    assert (head.exit_code, head.json["head"], checkpoints.exit_code, checkpoints.json) == (
        0,
        None,
        0,
        [],
    )


def test_fork_creates_a_trainable_model_from_its_source(hud: Hud, catalog: Platform) -> None:
    result = hud("models", "fork", "Claude Sonnet", "--name", "fresh", "--json")

    assert result.exit_code == 0, result
    (forked,) = [model for model in catalog.models if model["id"] == result.json["id"]]
    assert (forked["model_name"], forked["forked_from"]) == ("team/fresh", SONNET_ID)


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

    result = hud("models", "fork", "Claude Sonnet", "--name", "my-sonnet", *argv, "--json")

    assert result.exit_code == exit_code, result
    assert {key: result.json[key] for key in outcome} == outcome


@pytest.mark.parametrize(
    ("argv", "exit_code"),
    [
        pytest.param(["fork", "claude-sonnet-4-6", "--name", "mine", "--dry-run"], 0, id="dry-run"),
        pytest.param(["head", "gpt-nothing", "--set", "c"], 2, id="an-unknown-model"),
    ],
)
def test_dry_runs_and_unknown_models_change_nothing(
    hud: Hud, catalog: Platform, argv: list[str], exit_code: int
) -> None:
    result = hud("models", *argv, "--json")

    assert result.exit_code == exit_code, result
    assert len(catalog.models) == len(CATALOG)


def test_a_misspelt_model_suggests_the_close_name(hud: Hud, catalog: Platform) -> None:
    result = hud("models", "fork", "claude-sonet-4-6", "--name", "mine", "--json")

    assert (result.exit_code, result.json["error"]) == (2, "usage"), result
    assert "claude-sonnet-4-6" in result.json["message"]
