"""``hud jobs cancel``: it takes exactly one target."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.conftest import Hud
    from tests.fixtures.services import FakeServices

JOB_ID = "03dd2a73-d3df-4d10-a54a-e3d87c2d530d"
TRACE_ID = "00000000-0000-4000-a000-000000000002"


@pytest.mark.parametrize(
    "argv",
    [["jobs", "cancel"], ["jobs", "cancel", JOB_ID, "--all"]],
    ids=["nothing-named", "a-job-and-all"],
)
def test_cancel_needs_exactly_one_target(
    argv: list[str], signed_in: FakeServices, hud: Hud
) -> None:
    result = hud(*argv, "--yes", "--json")

    assert (result.exit_code, result.json["error"]) == (2, "usage"), result
    assert signed_in.requests("api", "POST") == []
