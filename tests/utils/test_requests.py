"""How the SDK talks to the HUD API: the platform client and the shared request policy."""

from __future__ import annotations

import time
import uuid
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from inline_snapshot import snapshot

from hud.utils import PlatformClient, make_request, make_request_sync
from hud.utils.exceptions import (
    HudAuthenticationError,
    HudDeprecationWarning,
    HudNetworkError,
    HudRequestError,
    HudTimeoutError,
)
from tests.harness import Reply

if TYPE_CHECKING:
    from tests.harness import FakeServices, Hud, HudEnv, Request


async def _request(asynchronous: bool, url: str, **kwargs: Any) -> Any:
    if asynchronous:
        return await make_request("GET", url, **kwargs)
    return make_request_sync("GET", url, **kwargs)


SYNC_AND_ASYNC = pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])


@SYNC_AND_ASYNC
async def test_transient_responses_are_retried_with_doubling_backoff(
    services: FakeServices, asynchronous: bool
) -> None:
    arrivals: list[float] = []
    replies = [Reply(status=503), Reply(status=502), Reply(json={"ok": True})]

    def answer(request: Request) -> Reply:
        del request
        arrivals.append(time.monotonic())
        return replies[len(arrivals) - 1]

    services.route("api", "GET", "/v2/data", handler=answer)

    result = await _request(
        asynchronous, f"{services.url('api')}/v2/data", api_key="k", retry_delay=0.1
    )

    assert result == {"ok": True}
    gaps = [later - earlier for earlier, later in pairwise(arrivals)]
    assert len(gaps) == 2
    assert gaps[0] >= 0.1
    assert gaps[1] >= 0.2


@SYNC_AND_ASYNC
@pytest.mark.parametrize(
    ("replies", "outcome", "attempts"),
    [
        pytest.param([Reply(json={"id": "1"})], {"id": "1"}, 1, id="ok"),
        pytest.param([Reply(status=204)], {}, 1, id="no-content"),
        pytest.param(
            [Reply(status=502), Reply(json={"id": "1"})], {"id": "1"}, 2, id="502-then-ok"
        ),
        pytest.param(
            [Reply(status=404, json={"detail": "nope"})],
            ("Request failed: nope", 404, {"detail": "nope"}),
            1,
            id="404-detail",
        ),
        pytest.param(
            [Reply(status=400, json={"code": "bad"})],
            (
                "Request failed with status 400 - JSON response: {'code': 'bad'}",
                400,
                {"code": "bad"},
            ),
            1,
            id="400-without-detail",
        ),
        pytest.param(
            [Reply(status=402, json={"detail": "no credits"})],
            ("Request failed: no credits", 402, {"detail": "no credits"}),
            1,
            id="402-once",
        ),
        pytest.param(
            [Reply(status=429, json={"detail": "slow down"})],
            ("Request failed: slow down", 429, {"detail": "slow down"}),
            1,
            id="429-once",
        ),
        pytest.param(
            [Reply(status=500, body="oops", content_type="text/plain")],
            ("Request failed with status 500", 500, None),
            1,
            id="500-once",
        ),
    ],
)
async def test_each_response_status_has_one_outcome(
    services: FakeServices,
    asynchronous: bool,
    replies: list[Reply],
    outcome: Any,
    attempts: int,
) -> None:
    services.route("api", "GET", "/v2/data", *replies)
    url = f"{services.url('api')}/v2/data"

    if isinstance(outcome, tuple):
        with pytest.raises(HudRequestError) as raised:
            await _request(asynchronous, url, api_key="k", retry_delay=0.01)
        message, status, body = outcome
        assert (raised.value.message, raised.value.status_code) == (message, status)
        assert raised.value.response_json == body
        assert f"Status: {status}" in str(raised.value)
    else:
        assert await _request(asynchronous, url, api_key="k", retry_delay=0.01) == outcome
    sent = services.requests("api", "GET", "/v2/data")
    assert len(sent) == attempts
    assert {request.headers["authorization"] for request in sent} == {"Bearer k"}


@SYNC_AND_ASYNC
@pytest.mark.parametrize(
    ("url", "kwargs", "error", "message"),
    [
        pytest.param(
            "{api}/v2/data",
            {"api_key": None},
            HudAuthenticationError,
            "API key is required",
            id="no-key",
        ),
        pytest.param(
            "http://127.0.0.1:9/v2/data",
            {"api_key": "k", "max_retries": 2, "retry_delay": 0.01},
            HudNetworkError,
            "Network error",
            id="unreachable",
        ),
        pytest.param(
            "{api}/v2/slow",
            {"api_key": "k"},
            HudTimeoutError,
            "Request timed out",
            id="timeout",
        ),
        pytest.param(
            "{api}/v2/not-json",
            {"api_key": "k"},
            HudRequestError,
            "Unexpected error",
            id="unparseable-body",
        ),
    ],
)
async def test_requests_that_never_get_an_answer_raise_typed_errors(
    services: FakeServices,
    asynchronous: bool,
    url: str,
    kwargs: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    services.route("api", "GET", "/v2/slow", Reply(json={}, delay=2.0))
    services.route("api", "GET", "/v2/not-json", Reply(body="<html>", content_type="text/html"))
    target = url.format(api=services.url("api"))
    timeout = httpx.Timeout(0.3)

    with pytest.raises(error, match=message):
        if asynchronous:
            async with httpx.AsyncClient(timeout=timeout) as client:
                await make_request("GET", target, client=client, **kwargs)
        else:
            with httpx.Client(timeout=timeout) as client:
                make_request_sync("GET", target, client=client, **kwargs)


def test_a_deprecation_notice_warns_once_per_process(services: FakeServices) -> None:
    successor = f"https://api.test/v2/{uuid.uuid4()}"
    headers = {
        "Deprecation": "@1790985600",
        "Sunset": "Sat, 10 Oct 2026 00:00:00 GMT",
        "Link": f'<{successor}>; rel="successor-version"',
    }
    services.route("api", "GET", "/v2/old/{id}", Reply(json={}, headers=headers))

    with pytest.warns(HudDeprecationWarning) as caught:
        for item in ("1", "2"):
            make_request_sync("GET", f"{services.url('api')}/v2/old/{item}", api_key="k")

    assert [str(warning.message) for warning in caught] == [
        "GET /api/v2/old/1 is deprecated by the HUD API and stops being served on "
        f"Sat, 10 Oct 2026 00:00:00 GMT; its replacement is {successor}. "
        "If a hud command printed this, upgrade hud."
    ]


# ─── the platform client ─────────────────────────────────────────────────────


async def test_the_platform_client_sends_each_verb_under_v2_with_the_bearer(
    services: FakeServices, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_API_KEY="k")
    for method in ("GET", "POST", "PUT", "PATCH", "DELETE"):
        services.route("api", method, "/v2/{rest:path}", json={"method": method})
    platform = PlatformClient.from_settings()

    answers = [
        platform.get("/tasks", params={"limit": 5, "ids": ["a", "b"]}),
        platform.post("/tasks", json={"b": 2}),
        platform.put("/tasks/1", json={"c": 3}),
        platform.patch("/tasks/1", json={"d": 4}),
        platform.delete("/tasks/1"),
        await platform.aget("/tasks", params={"limit": 1}),
        await platform.apost("/tasks", json={"e": 5}),
    ]

    assert [answer["method"] for answer in answers] == [
        "GET",
        "POST",
        "PUT",
        "PATCH",
        "DELETE",
        "GET",
        "POST",
    ]
    assert [
        (request.method, request.path, request.query, request.json, request.bearer)
        for request in services.requests("api")
    ] == snapshot(
        [
            ("GET", "/v2/tasks", {"limit": ["5"], "ids": ["a", "b"]}, None, "k"),
            ("POST", "/v2/tasks", {}, {"b": 2}, "k"),
            ("PUT", "/v2/tasks/1", {}, {"c": 3}, "k"),
            ("PATCH", "/v2/tasks/1", {}, {"d": 4}, "k"),
            ("DELETE", "/v2/tasks/1", {}, None, "k"),
            ("GET", "/v2/tasks", {"limit": ["1"]}, None, "k"),
            ("POST", "/v2/tasks", {}, {"e": 5}, "k"),
        ]
    )


@pytest.mark.parametrize("api_key", [None, ""])
def test_the_platform_client_needs_an_api_key(hud_env: HudEnv, api_key: str | None) -> None:
    hud_env.set(HUD_API_KEY=api_key)

    with pytest.raises(HudAuthenticationError, match="HUD_API_KEY is required"):
        PlatformClient.from_settings()


def test_a_platform_client_built_without_a_key_refuses_to_send(services: FakeServices) -> None:
    platform = PlatformClient(services.url("api"), "")

    with pytest.raises(HudAuthenticationError):
        platform.get("/tasks")
    assert services.requests() == []


# ─── through the CLI ─────────────────────────────────────────────────────────


def test_hud_models_list_reads_every_catalog_page(
    services: FakeServices, hud_env: HudEnv, hud: Hud
) -> None:
    hud_env.set(HUD_API_KEY="k")
    rows = [{"id": f"id-{index:03}", "name": f"Model {index:03}"} for index in range(103)]

    def page(request: Request) -> Reply:
        offset, limit = int(request.query["offset"][0]), int(request.query["limit"][0])
        return Reply(json={"items": rows[offset : offset + limit], "total": len(rows)})

    services.route("api", "GET", "/v2/models", handler=page)

    result = hud("models", "list", "--json")

    assert result.exit_code == 0, result
    assert [model["id"] for model in result.json] == [row["id"] for row in rows]
    assert [request.query for request in services.requests("api", "GET", "/v2/models")] == [
        {"limit": ["100"], "offset": ["0"]},
        {"limit": ["100"], "offset": ["100"]},
    ]


@pytest.mark.parametrize(
    ("reply", "exit_code", "document"),
    [
        pytest.param(
            Reply(status=404, json={"detail": "nope"}),
            1,
            snapshot(
                {
                    "error": "not_found",
                    "message": "Request failed: nope",
                    "suggestion": "Check the resource id, or list existing ones.",
                }
            ),
            id="not-found",
        ),
        pytest.param(
            Reply(status=429, json={"detail": "slow down"}),
            1,
            snapshot(
                {
                    "error": "rate_limited",
                    "message": "Request failed: slow down",
                    "suggestion": "Retry after a short delay.",
                }
            ),
            id="rate-limited",
        ),
        pytest.param(
            Reply(status=403, json={"detail": "forbidden"}),
            1,
            snapshot(
                {
                    "error": "permission_denied",
                    "message": "Request failed: forbidden",
                    "suggestion": "Check that this API key can access the resource.",
                }
            ),
            id="permission-denied",
        ),
    ],
)
def test_an_api_error_becomes_a_json_error_document(
    services: FakeServices,
    hud_env: HudEnv,
    hud: Hud,
    reply: Reply,
    exit_code: int,
    document: dict[str, Any],
) -> None:
    hud_env.set(HUD_API_KEY="k")
    services.route("api", "GET", "/v2/jobs", reply)

    result = hud("jobs", "list", "--json")

    assert result.exit_code == exit_code, result
    assert result.json == document
    assert len(services.requests("api", "GET", "/v2/jobs")) == 1


def test_a_deprecated_route_prints_one_warning_line(
    services: FakeServices, hud_env: HudEnv, hud: Hud
) -> None:
    hud_env.set(HUD_API_KEY="k")
    headers = {"Deprecation": "@1790985600", "Sunset": "Sat, 10 Oct 2026 00:00:00 GMT"}
    services.route("api", "GET", "/v2/jobs", Reply(json={"items": []}, headers=headers))

    result = hud("jobs", "list", "--json")

    assert result.exit_code == 0, result
    assert " ".join(result.stderr.split()) == (
        "⚠ GET /api/v2/jobs is deprecated by the HUD API and stops being served on "
        "Sat, 10 Oct 2026 00:00:00 GMT. If a hud command printed this, upgrade hud."
    )
