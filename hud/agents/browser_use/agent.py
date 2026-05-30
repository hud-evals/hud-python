"""BrowserUseAgent — delegates browser control to the ``browser-use`` SDK.

The env publishes a ``cdp/1.3`` capability (a Chromium DevTools endpoint); this
agent extracts that endpoint from the manifest and hands it to ``browser-use``,
which drives the browser over its own CDP client. We do **not** open one of our
own ``CapabilityClient`` connections — browser-use owns the session — so
``clients`` is empty and we only read the binding URL.

``browser-use`` is an optional dependency (``hud-python[browseruse]``); it is
imported lazily inside ``run`` so importing ``hud.agents`` never requires it.
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit, urlunsplit

from hud.agents.base import Agent
from hud.agents.types import BrowserUseConfig
from hud.settings import settings
from hud.types import Trace

if TYPE_CHECKING:
    from hud.client import Manifest

LOGGER = logging.getLogger("hud.agents.browser_use")

CDP_PROTOCOL = "cdp/1.3"


class BrowserUseAgent(Agent):
    """Run the ``browser-use`` agent against an env's ``cdp/1.3`` capability."""

    clients = ()  # browser-use owns its own CDP connection

    def __init__(self, config: BrowserUseConfig | None = None) -> None:
        self.config = config or BrowserUseConfig()
        self._cdp_url: str | None = None

    async def initialize(self, manifest: Manifest) -> None:
        await super().initialize(manifest)
        binding = next((b for b in manifest.bindings if b.protocol == CDP_PROTOCOL), None)
        if binding is None:
            raise ValueError("BrowserUseAgent requires a cdp/1.3 capability in the manifest")
        self._cdp_url = _ws_to_http(binding.url)
        LOGGER.info("browser-use will attach to %s", self._cdp_url)

    async def run(self, *, prompt: str, max_steps: int | None = None) -> Trace:
        if self._cdp_url is None:
            raise RuntimeError("initialize() must be called before run()")

        from browser_use import Agent as BrowserUseSdkAgent
        from browser_use import Browser, ChatAnthropic

        api_key = self.config.api_key or settings.anthropic_api_key
        if not api_key:
            raise ValueError("BrowserUseAgent needs an Anthropic API key (set ANTHROPIC_API_KEY)")

        llm = ChatAnthropic(model=self.config.model, api_key=api_key, base_url=self.config.base_url)
        browser: Any = Browser(cdp_url=self._cdp_url)
        sdk_agent = cast("Any", BrowserUseSdkAgent(task=prompt, llm=llm, browser=browser))

        try:
            history: Any = await sdk_agent.run(max_steps=max_steps or self.config.max_steps)
        except Exception as exc:
            LOGGER.exception("browser-use run failed")
            return Trace(done=True, content=str(exc), isError=True, info={"error": str(exc)})
        finally:
            with contextlib.suppress(Exception):
                await browser.stop()

        successful = history.is_successful()
        return Trace(
            done=history.is_done(),
            content=history.final_result() or "",
            isError=successful is False,
            info={
                "is_successful": successful,
                "steps": history.number_of_steps(),
                "urls": history.urls(),
            },
        )


def _ws_to_http(url: str) -> str:
    """Map a ``ws(s)://`` CDP endpoint to the ``http(s)://`` form browser-use expects."""
    parts = urlsplit(url)
    scheme = {"ws": "http", "wss": "https"}.get(parts.scheme, parts.scheme)
    return urlunsplit((scheme, parts.netloc, parts.path, parts.query, parts.fragment))


__all__ = ["BrowserUseAgent", "BrowserUseConfig"]
