"""``hud models``: the gateway catalog, forks, checkpoints and the served head."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from dirty_equals import IsStr
from inline_snapshot import snapshot

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


def test_models_list_shows_the_catalog_sorted_by_name(hud: Hud, catalog: FakeServices) -> None:
    result = hud("models", "list")

    assert result.exit_code == 0, result
    assert result.stdout.replace(catalog.url("gateway"), "<gateway>") == snapshot("""\
╭──────────────────╮
│ Available Models │
╰──────────────────╯
┏━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┓
┃           ┃ Model     ┃                                      ┃           ┃        ┃          ┃
┃ Name      ┃ (API)     ┃ ID                                   ┃ Provider  ┃ Agent  ┃ Trainab… ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━┩
│ Claude    │ claude-s… │ 00000000-0000-4000-a000-000000000001 │ anthropic │ claude │          │
│ Sonnet    │           │                                      │           │        │          │
│ owner/mo… │ team/my-… │ 00000000-0000-4000-a000-000000000002 │ anthropic │ claude │    ✓     │
│ + version │           │                                      │           │        │          │
└───────────┴───────────┴──────────────────────────────────────┴───────────┴────────┴──────────┘

Gateway: <gateway>
View a model in the browser: https://hud.example/models/<id>
""")
    (request,) = catalog.requests("api", "GET", "/v2/models")
    assert request.query == {"limit": ["100"], "offset": ["0"]}


@pytest.mark.parametrize(
    ("argv", "stdout"),
    [
        (["models", "list", "--quiet"], "claude-sonnet-4-6\nteam/my-sonnet\n"),
        (["models", "checkpoints", "my-sonnet", "--quiet"], "ckpt-0001-base\nckpt-0002-step\n"),
    ],
)
def test_quiet_prints_one_identifier_per_line(
    hud: Hud, catalog: FakeServices, argv: list[str], stdout: str
) -> None:
    result = hud(*argv)

    assert (result.exit_code, result.stdout) == (0, stdout)


def test_models_list_json_is_the_catalog_entries(hud: Hud, catalog: FakeServices) -> None:
    result = hud("models", "list", "--json")

    assert result.exit_code == 0, result
    assert result.json == snapshot(
        [
            {
                "id": "00000000-0000-4000-a000-000000000001",
                "name": "Claude Sonnet",
                "model_name": "claude-sonnet-4-6",
                "sdk_agent_type": "claude",
                "is_trainable": False,
                "provider": {"name": "anthropic"},
                "created_at": None,
                "released_at": None,
                "deprecated_at": None,
            },
            {
                "id": "00000000-0000-4000-a000-000000000002",
                "name": "owner/model + version",
                "model_name": "team/my-sonnet",
                "sdk_agent_type": "claude",
                "is_trainable": True,
                "provider": {"name": "anthropic"},
                "created_at": None,
                "released_at": None,
                "deprecated_at": None,
            },
        ]
    )


@pytest.mark.parametrize(
    ("argv", "requests"),
    [
        (["models", "checkpoints", "owner/model + version"], [("GET", "checkpoints", None)]),
        (["models", "head", "MY-SONNET"], [("GET", "checkpoints", None)]),
        (
            ["models", "head", MINE_ID, "--set", "ckpt-0001-base"],
            [("PUT", "head", {"checkpoint_id": "ckpt-0001-base"})],
        ),
        (["models", "head", "team/my-sonnet", "--set", "ckpt-0001-base", "--dry-run"], []),
    ],
)
def test_checkpoint_commands_address_the_resolved_model(
    hud: Hud, catalog: FakeServices, argv: list[str], requests: list[tuple[str, str, Any]]
) -> None:
    catalog.route("api", "PUT", "/v2/models/{id}/head", json={})

    text = hud(*argv)
    document = hud(*argv, "--json")

    assert text.exit_code == document.exit_code == 0, text
    assert (text.stdout, document.json) == RENDERED[" ".join(argv[1:])]
    sent = [
        (request.method, request.path.rsplit("/", 1)[-1], request.json)
        for request in catalog.requests("api", path=f"/v2/models/{MINE_ID}/{{verb}}")
    ]
    assert sent == requests * 2


RENDERED = snapshot(
    {
        "checkpoints owner/model + version": (
            """\
                                 Checkpoints                                 \n\
┏━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃   ┃ Name   ┃ Reward ┃ Loss                ┃ Traces ┃ Created              ┃
┡━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│   │ base   │      - │ -                   │      - │ 2026-09-01T00:00:00Z │
│ ▶ │ step-1 │  0.500 │ importance_sampling │     32 │ 2026-09-02T00:00:00Z │
└───┴────────┴────────┴─────────────────────┴────────┴──────────────────────┘

View: https://hud.example/models/00000000-0000-4000-a000-000000000002?tab=checkpoints
""",
            [
                {
                    "id": "ckpt-0001-base",
                    "name": "base",
                    "checkpoint_name": None,
                    "prev_model_checkpoint_id": None,
                    "is_active": False,
                    "created_at": "2026-09-01T00:00:00Z",
                    "num_traces": None,
                    "num_datums": None,
                    "num_tokens": None,
                    "mean_reward": None,
                    "learning_rate": None,
                    "loss_fn": None,
                    "metrics": {},
                },
                {
                    "id": "ckpt-0002-step",
                    "name": "step-1",
                    "checkpoint_name": "sampler-1",
                    "prev_model_checkpoint_id": None,
                    "is_active": True,
                    "created_at": "2026-09-02T00:00:00Z",
                    "num_traces": 32,
                    "num_datums": None,
                    "num_tokens": None,
                    "mean_reward": 0.5,
                    "learning_rate": None,
                    "loss_fn": "importance_sampling",
                    "metrics": {},
                },
            ],
        ),
        "head MY-SONNET": (
            """\
╭───────────────────────────────────────────────────────────╮
│ HEAD step-1                                               │
│ sampler: sampler-1                                        │
│ reward:  0.500    loss: importance_sampling    traces: 32 │
│ created: 2026-09-02T00:00:00Z                             │
╰───────────────────────────────────────────────────────────╯
View: https://hud.example/models/00000000-0000-4000-a000-000000000002?tab=checkpoints
""",
            {
                "model_id": "00000000-0000-4000-a000-000000000002",
                "head": {
                    "id": "ckpt-0002-step",
                    "name": "step-1",
                    "checkpoint_name": "sampler-1",
                    "prev_model_checkpoint_id": None,
                    "is_active": True,
                    "created_at": "2026-09-02T00:00:00Z",
                    "num_traces": 32,
                    "num_datums": None,
                    "num_tokens": None,
                    "mean_reward": 0.5,
                    "learning_rate": None,
                    "loss_fn": "importance_sampling",
                    "metrics": {},
                },
            },
        ),
        "head 00000000-0000-4000-a000-000000000002 --set ckpt-0001-base": (
            """\
Head set to ckpt-0001-base
View: https://hud.example/models/00000000-0000-4000-a000-000000000002?tab=checkpoints
""",
            {
                "model_id": "00000000-0000-4000-a000-000000000002",
                "checkpoint_id": "ckpt-0001-base",
                "action": "set_head",
            },
        ),
        "head team/my-sonnet --set ckpt-0001-base --dry-run": (
            "--dry-run: would set head of team/my-sonnet to ckpt-0001-base\n",
            {
                "dry_run": True,
                "action": "set_head",
                "model": "team/my-sonnet",
                "model_id": "00000000-0000-4000-a000-000000000002",
                "checkpoint_id": "ckpt-0001-base",
            },
        ),
    }
)


def test_a_model_with_no_checkpoints_serves_its_base_weights(
    hud: Hud, catalog: FakeServices
) -> None:
    catalog.route("api", "GET", "/v2/models/{id}/checkpoints", json=[])

    head = hud("models", "head", "claude-sonnet-4-6")
    checkpoints = hud("models", "checkpoints", "claude-sonnet-4-6")

    assert (head.exit_code, checkpoints.exit_code) == (0, 0)
    assert head.lines + checkpoints.lines == snapshot(
        [
            "No active checkpoint — this model serves its base weights",
            "View: https://hud.example/models/00000000-0000-4000-a000-000000000001?tab=checkpoints",
            "No checkpoints yet — this model serves its base weights",
            "View: https://hud.example/models/00000000-0000-4000-a000-000000000001?tab=checkpoints",
        ]
    )


@pytest.mark.parametrize(
    ("argv", "replies", "exit_code", "document"),
    [
        (
            ["models", "fork", "claude-sonnet-4-6", "--name", "my-sonnet"],
            [Reply(json={"id": MINE_ID, "name": "my-sonnet", "model_name": "team/my-sonnet"})],
            0,
            snapshot(
                {
                    "id": "00000000-0000-4000-a000-000000000002",
                    "name": "my-sonnet",
                    "model_name": "team/my-sonnet",
                }
            ),
        ),
        (
            ["models", "fork", "Claude Sonnet", "--name", "my-sonnet", "--if-not-exists"],
            [Reply(status=409, json={"detail": "Model exists"})],
            0,
            snapshot(
                {
                    "id": "00000000-0000-4000-a000-000000000002",
                    "name": "owner/model + version",
                    "model_name": "team/my-sonnet",
                    "sdk_agent_type": "claude",
                    "is_trainable": True,
                    "provider": {"name": "anthropic"},
                    "created_at": None,
                    "released_at": None,
                    "deprecated_at": None,
                    "existed": True,
                }
            ),
        ),
        (
            ["models", "fork", "claude-sonnet-4-6", "--name", "my-sonnet"],
            [Reply(status=409, json={"detail": "Model exists"})],
            1,
            snapshot(
                {
                    "error": "conflict",
                    "message": "Request failed: Model exists",
                    "input": {"source": "claude-sonnet-4-6", "name": "my-sonnet"},
                }
            ),
        ),
    ],
)
def test_fork_creates_a_trainable_model(
    hud: Hud,
    catalog: FakeServices,
    argv: list[str],
    replies: list[Reply],
    exit_code: int,
    document: dict[str, Any],
) -> None:
    catalog.route("api", "POST", "/v2/models/fork", *replies)

    result = hud(*argv, "--json")

    assert result.exit_code == exit_code, result
    assert result.json == document
    assert catalog.bodies("api", "POST", "/v2/models/fork") == [
        {"source_model_id": SONNET_ID, "name": "my-sonnet"}
    ]


def test_fork_prints_how_to_train_the_new_model(hud: Hud, catalog: FakeServices) -> None:
    catalog.route(
        "api", "POST", "/v2/models/fork", json={"id": MINE_ID, "model_name": "team/my-sonnet"}
    )

    result = hud("models", "fork", "claude-sonnet-4-6", "--name", "my-sonnet")

    assert result.exit_code == 0, result
    assert result.lines == snapshot(
        [
            "Forked team/my-sonnet",
            "slug: team/my-sonnet",
            "id: 00000000-0000-4000-a000-000000000002",
            "Train it: hud.TrainingClient('team/my-sonnet')",
            "View: https://hud.example/models/00000000-0000-4000-a000-000000000002",
        ]
    )


@pytest.mark.parametrize(
    ("argv", "exit_code", "document"),
    [
        (
            ["models", "fork", "claude-sonnet-4-6", "--name", "mine", "--dry-run"],
            0,
            snapshot(
                {
                    "dry_run": True,
                    "action": "fork",
                    "source": "claude-sonnet-4-6",
                    "name": "mine",
                    "if_not_exists": False,
                }
            ),
        ),
        (
            ["models", "fork", "claude-sonet-4-6", "--name", "mine"],
            2,
            {
                "error": "usage",
                "message": IsStr(
                    regex=r"Model 'claude-sonet-4-6' not found in the HUD gateway registry\. "
                    r"Did you mean: .*claude-sonnet-4-6.*\?"
                ),
            },
        ),
        (
            ["models", "head", "gpt-nothing"],
            2,
            {
                "error": "usage",
                "message": "Model 'gpt-nothing' not found in the HUD gateway registry. "
                "Run `hud models` to list them.",
            },
        ),
    ],
)
def test_fork_dry_runs_and_unknown_models_make_no_change(
    hud: Hud, catalog: FakeServices, argv: list[str], exit_code: int, document: dict[str, Any]
) -> None:
    result = hud(*argv, "--json")

    assert result.exit_code == exit_code, result
    assert result.json == document
    assert catalog.requests("api", "POST") + catalog.requests("api", "PUT") == []
