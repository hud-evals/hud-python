"""The shared request policy where the CLI cannot see it: retry timing, the attempts a
failure gets, timeouts, and deprecation notices.

How each platform answer reaches a user as an error kind is ``tests/cli/test_output.py``.
"""

from __future__ import annotations

import time
import uuid
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from hud.utils import make_request, make_request_sync
from hud.utils.exceptions import (
    HudDeprecationWarning,
    HudNetworkError,
    HudRequestError,
    HudTimeoutError,
)
from tests.support import Reply

if TYPE_CHECKING:
    from tests.support import FakeServices, Request


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
        pytest.param([Reply(status=204)], {}, 1, id="no-content"),
        pytest.param(
            [Reply(status=500, body="oops", content_type="text/plain")],
            ("Request failed with status 500", 500),
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
        assert (raised.value.message, raised.value.status_code) == outcome
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

    (warning,) = caught
    assert all(
        part in str(warning.message) for part in ("Sat, 10 Oct 2026 00:00:00 GMT", successor)
    )
