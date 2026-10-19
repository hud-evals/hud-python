"""What only ``BrowserUseAgent`` does: hand browser-use the run's task and its browser, and
fold the history browser-use returns into the trace.

browser-use is the outside world here: the stand-in SDK (``tests/support/browser_use.py``)
lists the browser's pages over the DevTools HTTP endpoint it is handed and asks
the scripted model until it replies ``done: <answer>``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from dirty_equals import IsStr

from hud.agents.types import AgentStep
from tests.agents.families import BrowserUseDriver
from tests.support import say

if TYPE_CHECKING:
    from types import ModuleType

    from tests.agents.families import Driver

pytestmark = pytest.mark.parametrize("family", ["browser_use"])

HISTORIES = {
    "finished": ("done: blue", ("blue", True, 1, True)),
    "out-of-steps": ("still looking", ("", False, 2, False)),
}


@pytest.mark.parametrize(("reply", "expected"), HISTORIES.values(), ids=HISTORIES.keys())
async def test_browser_use_gets_the_task_and_the_browser_and_its_history_lands_on_the_trace(
    reply: str,
    expected: tuple[str, bool, int, bool],
    family: str,
    driver: Driver,
    browser_use_sdk: ModuleType,
) -> None:
    assert isinstance(driver, BrowserUseDriver)
    driver.provider.respond(lambda _request: say(reply))

    run = await driver.run(driver.agent(max_steps=2), prompt="What colour is the sky?")

    (seen, *_) = driver.seen()
    (step,) = [step for step in run.trace.steps if isinstance(step, AgentStep)]
    extra = run.trace.extra
    assert (seen.prompt, driver.browser.http) == ("What colour is the sky?", ["GET /json"])
    assert browser_use_sdk.stopped == [IsStr(regex=r"http://127\.0\.0\.1:\d+")]
    assert (run.trace.content, extra["is_successful"], extra["steps"], step.done) == expected
    assert extra["urls"] == ["page-1"]
