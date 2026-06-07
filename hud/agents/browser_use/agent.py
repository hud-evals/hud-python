"""BrowserUseAgent — delegates browser control to the ``browser-use`` SDK.

The env publishes a ``cdp/1.3`` capability (a Chromium DevTools endpoint); this
agent reads that binding off the run's manifest and hands the URL to
``browser-use``, which drives the browser over its own CDP client. We do **not**
``open`` one of our own ``CapabilityClient`` connections — browser-use owns the
session — so this agent reaches for ``trace.binding(...)`` (raw declaration)
rather than ``trace.open(...)`` (managed client).

The agent is stateless w.r.t. the env: it holds only config and is driven by
``await agent(run)``, receiving the run handle per call. ``browser-use`` is an
optional dependency
(``hud-python[browseruse]``), imported lazily inside ``rollout``.
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit, urlunsplit

from hud.agents.base import Agent
from hud.agents.types import BrowserUseConfig
from hud.settings import settings

if TYPE_CHECKING:
    from hud.client import Run

LOGGER = logging.getLogger("hud.agents.browser_use")

CDP_PROTOCOL = "cdp/1.3"


class BrowserUseAgent(Agent):
    """Run the ``browser-use`` agent against an env's ``cdp/1.3`` capability."""

    def __init__(self, config: BrowserUseConfig | None = None) -> None:
        self.config = config or BrowserUseConfig()

    async def __call__(self, run: Run) -> None:
        """Drive browser-use over the run's CDP capability, filling ``run.trace``.

        Reads ``run.prompt`` and the CDP binding off the run, runs the browser-use
        loop, and writes the final answer + trajectory metadata onto ``run.trace``
        (graded on exit).
        """
        from browser_use import Agent as BrowserUseSdkAgent
        from browser_use import Browser, ChatAnthropic

        trace = run.trace
        cdp_url = _ws_to_http(run.client.binding(CDP_PROTOCOL).url)
        LOGGER.info("browser-use attaching to %s", cdp_url)

        api_key = self.config.api_key or settings.anthropic_api_key
        if not api_key:
            raise ValueError("BrowserUseAgent needs an Anthropic API key (set ANTHROPIC_API_KEY)")

        llm = ChatAnthropic(model=self.config.model, api_key=api_key, base_url=self.config.base_url)
        browser: Any = Browser(cdp_url=cdp_url)
        sdk_agent = cast("Any", BrowserUseSdkAgent(task=run.prompt or "", llm=llm, browser=browser))

        try:
            history: Any = await sdk_agent.run(max_steps=self.config.max_steps)
        except Exception as exc:
            LOGGER.exception("browser-use run failed")
            trace.done = True
            trace.content = str(exc)
            trace.isError = True
            trace.info["error"] = str(exc)
            return
        finally:
            with contextlib.suppress(Exception):
                await browser.stop()

        successful = history.is_successful()
        trace.done = history.is_done()
        trace.content = history.final_result() or ""
        trace.isError = successful is False
        trace.info.update(
            {
                "is_successful": successful,
                "steps": history.number_of_steps(),
                "urls": history.urls(),
            }
        )


def _ws_to_http(url: str) -> str:
    """Map a ``ws(s)://`` CDP endpoint to the ``http(s)://`` form browser-use expects."""
    parts = urlsplit(url)
    scheme = {"ws": "http", "wss": "https"}.get(parts.scheme, parts.scheme)
    return urlunsplit((scheme, parts.netloc, parts.path, parts.query, parts.fragment))


__all__ = ["BrowserUseAgent", "BrowserUseConfig"]
