from __future__ import annotations

import asyncio
from functools import partial
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from hud.settings import settings
from hud.telemetry.context import set_trace_context
from hud.utils import gateway
from hud.utils.exceptions import HudAuthenticationError

if TYPE_CHECKING:
    from google.genai import Client as GenaiClient
    from openai import AsyncOpenAI


@pytest.fixture(autouse=True)
def _gateway_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_key", "sk-hud-test")
    monkeypatch.setattr(settings, "hud_gateway_url", "https://gateway.test")


@pytest.mark.asyncio
async def test_openai_client_resolves_trace_id_per_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_headers: dict[str, str] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        trace_id = request.headers["Trace-Id"]
        await asyncio.sleep(0)
        seen_headers[request.url.path] = trace_id
        return httpx.Response(200, json={"object": "list", "data": []})

    transport = httpx.MockTransport(handler)
    real_client_factory = gateway.DefaultAsyncHttpxClient
    monkeypatch.setattr(
        gateway,
        "DefaultAsyncHttpxClient",
        partial(real_client_factory, transport=transport),
    )
    client = cast("AsyncOpenAI", gateway.build_gateway_client("openai"))

    async def request_in_trace(trace_id: str) -> None:
        with set_trace_context(trace_id):
            await client.get(f"/models/{trace_id}", cast_to=object)

    try:
        await asyncio.gather(
            request_in_trace("11111111-1111-4111-8111-111111111111"),
            request_in_trace("22222222-2222-4222-8222-222222222222"),
        )
    finally:
        await client.close()

    assert seen_headers == {
        "/models/11111111-1111-4111-8111-111111111111": ("11111111-1111-4111-8111-111111111111"),
        "/models/22222222-2222-4222-8222-222222222222": ("22222222-2222-4222-8222-222222222222"),
    }


@pytest.mark.asyncio
async def test_openai_client_trace_context_overrides_empty_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_trace_id: str | None = None

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_trace_id
        seen_trace_id = request.headers["Trace-Id"]
        return httpx.Response(200, json={"object": "list", "data": []})

    transport = httpx.MockTransport(handler)
    real_client_factory = gateway.DefaultAsyncHttpxClient
    monkeypatch.setattr(
        gateway,
        "DefaultAsyncHttpxClient",
        partial(real_client_factory, transport=transport),
    )
    client = cast("AsyncOpenAI", gateway.build_gateway_client("openai"))
    trace_id = "11111111-1111-4111-8111-111111111111"

    try:
        with set_trace_context(trace_id):
            await client.get(
                "/models/explicit",
                cast_to=object,
                options={"headers": {"Trace-Id": ""}},
            )
    finally:
        await client.close()

    assert seen_trace_id == trace_id


@pytest.mark.asyncio
async def test_openai_client_sends_child_and_parent_trace_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_headers: dict[str, str] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_headers["Trace-Id"] = request.headers["Trace-Id"]
        seen_headers["X-HUD-Parent-Trace-Id"] = request.headers["X-HUD-Parent-Trace-Id"]
        return httpx.Response(200, json={"object": "list", "data": []})

    transport = httpx.MockTransport(handler)
    real_client_factory = gateway.DefaultAsyncHttpxClient
    monkeypatch.setattr(
        gateway,
        "DefaultAsyncHttpxClient",
        partial(real_client_factory, transport=transport),
    )
    client = cast("AsyncOpenAI", gateway.build_gateway_client("openai"))

    try:
        with set_trace_context("child", parent_trace_id="parent"):
            await client.get("/models/nested", cast_to=object)
    finally:
        await client.close()

    assert seen_headers == {
        "Trace-Id": "child",
        "X-HUD-Parent-Trace-Id": "parent",
    }


_BEDROCK_ARN = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/anthropic.claude"


def test_list_gateway_models_reads_every_page(monkeypatch: pytest.MonkeyPatch) -> None:
    gateway.list_gateway_models.cache_clear()
    rows = [{"id": f"m{i}", "model_name": f"m{i}"} for i in range(103)]

    def get(path: str, *, params: dict[str, int]) -> dict[str, object]:
        assert path == "/models"
        start = params["offset"]
        return {"items": rows[start : start + params["limit"]], "total": len(rows)}

    platform = MagicMock()
    platform.get.side_effect = get
    monkeypatch.setattr(gateway.PlatformClient, "from_settings", lambda: platform)
    try:
        models = gateway.list_gateway_models()
    finally:
        gateway.list_gateway_models.cache_clear()

    assert [m.model_name for m in models] == [f"m{i}" for i in range(103)]
    assert platform.get.call_count == 2


def test_model_recency_prefers_release_over_catalog_date() -> None:
    released = gateway.GatewayModelInfo(
        id="released", created_at="2026-09-01T00:00:00Z", released_at="2026-01-01T00:00:00Z"
    )
    added = gateway.GatewayModelInfo(id="added", created_at="2026-06-01T00:00:00Z")
    undated = gateway.GatewayModelInfo(id="undated")
    assert sorted([added, released, undated], key=lambda m: m.recency) == [
        undated,
        released,
        added,
    ]


def test_bedrock_arn_model_gets_a_bedrock_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "aws_access_key_id", "AKIATEST")
    monkeypatch.setattr(settings, "aws_secret_access_key", "secret")
    monkeypatch.setattr(settings, "aws_region", "us-east-1")
    bedrock = MagicMock(return_value=object())
    monkeypatch.setattr("anthropic.AsyncAnthropicBedrock", bedrock)
    gateway_client = MagicMock()
    monkeypatch.setattr(gateway, "build_gateway_client", gateway_client)

    client = gateway.build_model_client("anthropic", model=_BEDROCK_ARN)

    assert client is bedrock.return_value
    bedrock.assert_called_once_with(
        aws_access_key="AKIATEST", aws_secret_key="secret", aws_region="us-east-1"
    )
    gateway_client.assert_not_called()


def test_bedrock_arn_model_requires_aws_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "aws_access_key_id", None)
    monkeypatch.setattr(settings, "aws_secret_access_key", None)
    monkeypatch.setattr(settings, "aws_region", None)

    with pytest.raises(HudAuthenticationError, match="AWS Bedrock"):
        gateway.build_model_client("anthropic", model=_BEDROCK_ARN)


def test_bedrock_arn_model_cannot_be_forced_through_the_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "aws_access_key_id", "AKIATEST")
    monkeypatch.setattr(settings, "aws_secret_access_key", "secret")
    monkeypatch.setattr(settings, "aws_region", "us-east-1")
    bedrock = MagicMock()
    monkeypatch.setattr("anthropic.AsyncAnthropicBedrock", bedrock)

    with pytest.raises(ValueError, match="cannot use the HUD gateway"):
        gateway.build_model_client("anthropic", model=_BEDROCK_ARN, gateway=True)
    bedrock.assert_not_called()


def test_provider_key_wins_over_hud_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    direct = MagicMock(return_value=object())
    monkeypatch.setattr("anthropic.AsyncAnthropic", direct)
    gateway_client = MagicMock()
    monkeypatch.setattr(gateway, "build_gateway_client", gateway_client)

    client = gateway.build_model_client("anthropic", model="claude-sonnet-4-6")

    assert client is direct.return_value
    direct.assert_called_once_with(api_key="sk-ant-test")
    gateway_client.assert_not_called()


def test_hud_key_alone_routes_through_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "anthropic_api_key", None)
    gateway_client = MagicMock(return_value=object())
    monkeypatch.setattr(gateway, "build_gateway_client", gateway_client)

    client = gateway.build_model_client("anthropic", model="claude-sonnet-4-6")

    assert client is gateway_client.return_value
    gateway_client.assert_called_once_with("anthropic")


def test_no_key_at_all_is_an_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_key", None)
    monkeypatch.setattr(settings, "openai_api_key", None)
    with pytest.raises(HudAuthenticationError, match="No API key for openai"):
        gateway.build_model_client("openai")


@pytest.mark.asyncio
async def test_anthropic_client_receives_trace_aware_http_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client_factory = MagicMock(return_value=object())
    monkeypatch.setattr("anthropic.AsyncAnthropic", client_factory)

    gateway.build_gateway_client("anthropic")

    kwargs = client_factory.call_args.kwargs
    http_client = kwargs["http_client"]
    assert http_client.event_hooks["request"]
    await http_client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_status", [200, 429, 503])
async def test_gemini_async_request_includes_trace_id(
    monkeypatch: pytest.MonkeyPatch,
    initial_status: int,
) -> None:
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep)
    seen_trace_id: str | None = None
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_trace_id, attempts
        seen_trace_id = request.headers["Trace-Id"]
        attempts += 1
        if attempts == 1 and initial_status != 200:
            return httpx.Response(
                initial_status, json={"error": {"code": initial_status, "message": "busy"}}
            )
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"parts": [{"text": "ok"}], "role": "model"},
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ],
                "modelVersion": "gemini-test",
                "usageMetadata": {
                    "candidatesTokenCount": 1,
                    "promptTokenCount": 1,
                    "totalTokenCount": 2,
                },
            },
        )

    monkeypatch.setattr(
        gateway.httpx,
        "AsyncHTTPTransport",
        MagicMock(return_value=httpx.MockTransport(handler)),
    )
    client = cast("GenaiClient", gateway.build_gateway_client("gemini"))
    trace_id = "11111111-1111-4111-8111-111111111111"

    try:
        with set_trace_context(trace_id):
            response = await client.aio.models.generate_content(
                model="gemini-test",
                contents="hi",
            )
    finally:
        await client.aio.aclose()

    assert response.text == "ok"
    assert seen_trace_id == trace_id
    assert attempts == (1 if initial_status == 200 else 2)
    if initial_status != 200:
        sleep.assert_awaited_once()
        assert 10 <= sleep.await_args_list[0].args[0] <= 11
    else:
        sleep.assert_not_awaited()
