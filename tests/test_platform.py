"""``PlatformClient``, which environments and scripts call directly: a request that names no
reply type gets the decoded JSON, as it always has."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hud.utils.platform import PlatformClient
from tests.fixtures.services import Reply

if TYPE_CHECKING:
    from tests.conftest import HudEnv
    from tests.fixtures.services import FakeServices

TASKSET = {"id": "eeeeeeee-0000-4000-8000-000000000001", "can_edit": True, "extra": [1, 2]}


async def test_a_request_without_a_reply_type_returns_the_decoded_json(
    services: FakeServices, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_API_KEY="k")
    services.route("api", "GET", "/v2/tasksets/{id}", json=TASKSET)
    services.route("api", "POST", "/v2/data/{id}/complete", Reply(status=204, body=b""))
    platform = PlatformClient.from_settings()

    assert await platform.aget(f"/tasksets/{TASKSET['id']}") == TASKSET
    assert platform.get(f"/tasksets/{TASKSET['id']}") == TASKSET
    assert await platform.apost("/data/f/complete") == {}
