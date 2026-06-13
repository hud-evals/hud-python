"""BrowserUseAgent — delegates browser control to the ``browser-use`` SDK.

The env publishes a ``cdp/1.3`` capability (a Chromium DevTools endpoint); this
agent reads that binding's URL and hands it to ``browser-use``, which drives
the browser over its own CDP client. We do **not** ``open`` one of our own
``CapabilityClient`` connections — browser-use owns the session — so this
agent uses ``client.binding(...)`` (wire data) rather than ``client.open(...)``
(managed client).

The agent is stateless w.r.t. the env: it holds only config and is driven by
``await agent(run)``, receiving the run handle per call. ``browser-use`` is an
optional dependency
(``hud-python[browseruse]``), imported lazily inside ``rollout``.
"""

from __future__ import annotations

# browser-use is an optional, untyped dependency (lazy __getattr__ exports), so
# its symbols and members resolve as Unknown under strict checking. This whole
# module is the boundary around that SDK; contain the unknowns here.
# pyright: reportAttributeAccessIssue=false, reportUnknownVariableType=false, reportUnknownMemberType=false
import contextlib
import logging
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit, urlunsplit

from hud.agents.base import Agent
from hud.agents.types import AgentStep, BrowserUseConfig
from hud.settings import settings
from hud.types import Step

if TYPE_CHECKING:
    from hud.eval.run import Run

LOGGER = logging.getLogger("hud.agents.browser_use")

CDP_PROTOCOL = "cdp/1.3"


class BrowserUseAgent(Agent):
    """Run the ``browser-use`` agent against an env's ``cdp/1.3`` capability."""

    config: BrowserUseConfig

    def __init__(self, config: BrowserUseConfig | None = None) -> None:
        self.config = config or BrowserUseConfig()

    async def __call__(self, run: Run) -> None:
        """Drive browser-use over the run's CDP capability, filling ``run.trace``.

        Reads ``run.prompt_text`` and the CDP binding off the run, runs the browser-use
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
        sdk_agent = cast("Any", BrowserUseSdkAgent(task=run.prompt_text, llm=llm, browser=browser))

        try:
            history: Any = await sdk_agent.run(max_steps=self.config.max_steps)
        except Exception as exc:
            LOGGER.exception("browser-use run failed")
            trace.status = "error"
            run.record(Step(source="system", error=str(exc)))
            return
        finally:
            with contextlib.suppress(Exception):
                await browser.stop()

        successful = history.is_successful()
        content = history.final_result() or ""
        trace.status = "error" if successful is False else "completed"
        trace.content = content
        trace.extra.update(
            {
                "is_successful": successful,
                "steps": history.number_of_steps(),
                "urls": history.urls(),
            }
        )
        # browser-use owns its own loop; record the run as one coarse agent step
        # (per-action fidelity would need a browser-use-native serializer).
        run.record(
            AgentStep(
                content=content,
                done=history.is_done(),
                error=content if successful is False else None,
            ),
        )


def _ws_to_http(url: str) -> str:
    """Map a ``ws(s)://`` CDP endpoint to the ``http(s)://`` form browser-use expects."""
    parts = urlsplit(url)
    scheme = {"ws": "http", "wss": "https"}.get(parts.scheme, parts.scheme)
    return urlunsplit((scheme, parts.netloc, parts.path, parts.query, parts.fragment))


__all__ = ["BrowserUseAgent", "BrowserUseConfig"]
