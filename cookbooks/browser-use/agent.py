"""browser-use as a HUD agent: it drives the browser an environment publishes as ``cdp``.

The agent reads the ``cdp/1.3`` binding's URL off the run and hands it to browser-use,
which drives the browser over its own CDP client and runs its own loop. The run
records one coarse agent step with browser-use's final answer.
"""

from __future__ import annotations

import contextlib
import os
from urllib.parse import urlsplit, urlunsplit

from browser_use import Agent as BrowserUse
from browser_use import Browser, ChatAnthropic

from hud.agents.base import Agent
from hud.agents.types import AgentConfig, AgentStep
from hud import Run


class BrowserUseConfig(AgentConfig):
    model: str = "claude-sonnet-4-5"
    max_steps: int = 25


class BrowserUseAgent(Agent[BrowserUseConfig]):
    config_cls = BrowserUseConfig

    async def __call__(self, run: Run) -> None:
        cdp = urlsplit(run.client.binding("cdp/1.3").url)
        scheme = {"ws": "http", "wss": "https"}.get(cdp.scheme, cdp.scheme)
        browser = Browser(cdp_url=urlunsplit((scheme, *cdp[1:])))
        llm = ChatAnthropic(model=self.config.model, api_key=os.environ["ANTHROPIC_API_KEY"])
        try:
            history = await BrowserUse(task=run.prompt_text, llm=llm, browser=browser).run(
                max_steps=self.config.max_steps
            )
        finally:
            with contextlib.suppress(Exception):
                await browser.stop()

        run.trace.content = history.final_result() or ""
        run.trace.extra.update(
            {
                "is_successful": history.is_successful(),
                "steps": history.number_of_steps(),
                "urls": history.urls(),
            }
        )
        run.record(AgentStep(content=run.trace.content, done=history.is_done()))
