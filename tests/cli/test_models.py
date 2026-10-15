"""``hud models``: the gateway catalog, forks, checkpoints and the served head."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from tests.harness import Reply

if TYPE_CHECKING:
    from tests.harness import FakeServices, Hud

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
def catalog(platform: FakeServices) -> FakeServices:
    platform.route("api", "GET", "/v2/models", json={"items": CATALOG, "total": len(CATALOG)})
    platform.route("api", "GET", "/v2/models/{id}/checkpoints", json=CHECKPOINTS)
    return platform


@pytest.mark.parametrize(
    ("argv", "stdout"),
    [
        (["models", "list", "--quiet"], ["claude-sonnet-4-6", "team/my-sonnet"]),
        (["models", "checkpoints", "my-sonnet", "--quiet"], ["ckpt-0001-base", "ckpt-0002-step"]),
    ],
)
def test_quiet_prints_one_identifier_per_line(
    hud: Hud, catalog: FakeServices, argv: list[str], stdout: list[str]
) -> None:
    result = hud(*argv)

    assert (result.exit_code, result.stdout.split("\n")[:-1]) == (0, stdout), result


def test_models_list_json_is_the_catalog_sorted_by_name(hud: Hud, catalog: FakeServices) -> None:
    result = hud("models", "list", "--json")

    assert result.exit_code == 0, result
    assert [model["id"] for model in result.json] == [SONNET_ID, MINE_ID]
    (request,) = catalog.requests("api", "GET", "/v2/models")
    assert request.query == {"limit": ["100"], "offset": ["0"]}


@pytest.mark.parametrize("model", ["owner/model + version", "MY-SONNET", "team/my-sonnet", MINE_ID])
def test_checkpoints_and_head_read_the_resolved_model(
    hud: Hud, catalog: FakeServices, model: str
) -> None:
    checkpoints = hud("models", "checkpoints", model, "--json")
    head = hud("models", "head", model, "--json")

    assert (checkpoints.exit_code, head.exit_code) == (0, 0), head
    assert [checkpoint["id"] for checkpoint in checkpoints.json] == [
        "ckpt-0001-base",
        "ckpt-0002-step",
    ]
    assert (head.json["model_id"], head.json["head"]["id"]) == (MINE_ID, "ckpt-0002-step")
    assert {r.path for r in catalog.requests("api", "GET", "/v2/models/{id}/checkpoints")} == {
        f"/v2/models/{MINE_ID}/checkpoints"
    }


@pytest.mark.parametrize(("argv", "set_to"), [([], ["ckpt-0001-base"]), (["--dry-run"], [])])
def test_head_set_moves_the_served_checkpoint(
    hud: Hud, catalog: FakeServices, argv: list[str], set_to: list[str]
) -> None:
    catalog.route("api", "PUT", "/v2/models/{id}/head", json={})

    result = hud("models", "head", "my-sonnet", "--set", "ckpt-0001-base", *argv, "--json")

    assert (result.exit_code, result.json["checkpoint_id"]) == (0, "ckpt-0001-base"), result
    assert [
        body["checkpoint_id"] for body in catalog.bodies("api", "PUT", f"/v2/models/{MINE_ID}/head")
    ] == set_to


def test_a_model_with_no_checkpoints_has_no_head(hud: Hud, catalog: FakeServices) -> None:
    catalog.route("api", "GET", "/v2/models/{id}/checkpoints", json=[])

    head = hud("models", "head", "claude-sonnet-4-6", "--json")
    checkpoints = hud("models", "checkpoints", "claude-sonnet-4-6", "--json")

    assert (head.exit_code, head.json["head"], checkpoints.exit_code, checkpoints.json) == (
        0,
        None,
        0,
        [],
    )


@pytest.mark.parametrize(
    ("argv", "reply", "exit_code", "outcome"),
    [
        pytest.param(
            [],
            Reply(json={"id": MINE_ID, "name": "my-sonnet", "model_name": "team/my-sonnet"}),
            0,
            {"id": MINE_ID},
            id="created",
        ),
        pytest.param(
            ["--if-not-exists"],
            Reply(status=409, json={"detail": "Model exists"}),
            0,
            {"id": MINE_ID, "existed": True},
            id="already-there-and-allowed",
        ),
        pytest.param(
            [],
            Reply(status=409, json={"detail": "Model exists"}),
            1,
            {"error": "conflict"},
            id="already-there",
        ),
    ],
)
def test_fork_creates_a_trainable_model(
    hud: Hud,
    catalog: FakeServices,
    argv: list[str],
    reply: Reply,
    exit_code: int,
    outcome: dict[str, Any],
) -> None:
    catalog.route("api", "POST", "/v2/models/fork", reply)

    result = hud("models", "fork", "Claude Sonnet", "--name", "my-sonnet", *argv, "--json")

    assert result.exit_code == exit_code, result
    assert {key: result.json[key] for key in outcome} == outcome
    assert catalog.bodies("api", "POST", "/v2/models/fork") == [
        {"source_model_id": SONNET_ID, "name": "my-sonnet"}
    ]


@pytest.mark.parametrize(
    ("argv", "exit_code"),
    [
        pytest.param(["fork", "claude-sonnet-4-6", "--name", "mine", "--dry-run"], 0, id="dry-run"),
        pytest.param(["fork", "claude-sonet-4-6", "--name", "mine"], 2, id="a-misspelt-model"),
        pytest.param(["head", "gpt-nothing", "--set", "c"], 2, id="an-unknown-model"),
    ],
)
def test_dry_runs_and_unknown_models_change_nothing(
    hud: Hud, catalog: FakeServices, argv: list[str], exit_code: int
) -> None:
    result = hud("models", *argv, "--json")

    assert result.exit_code == exit_code, result
    assert catalog.requests("api", "POST") + catalog.requests("api", "PUT") == []


def test_a_misspelt_model_suggests_the_close_name(hud: Hud, catalog: FakeServices) -> None:
    result = hud("models", "fork", "claude-sonet-4-6", "--name", "mine", "--json")

    assert (result.exit_code, result.json["error"]) == (2, "usage"), result
    assert "claude-sonnet-4-6" in result.json["message"]
