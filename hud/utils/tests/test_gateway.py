from __future__ import annotations

import asyncio
from functools import partial
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import httpx
import pytest

from hud.settings import settings
from hud.telemetry.context import set_trace_context
from hud.utils import gateway

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
async def test_gemini_async_request_includes_trace_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_trace_id: str | None = None

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_trace_id
        seen_trace_id = request.headers["Trace-Id"]
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
