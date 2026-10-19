"""A stand-in for the ``browser-use`` SDK, the outside world ``BrowserUseAgent`` hands a run to.

It keeps the shape the agent drives: ``ChatAnthropic`` (an Anthropic Messages
client), ``Browser`` (a Chromium DevTools endpoint) and ``Agent.run(max_steps=)``
returning a history. ``Agent.run`` lists the browser's page targets over
``GET /json``, then asks the model once per step until it replies ``done: <answer>``
or the steps run out. The ``browser_use_sdk`` fixture puts it on ``sys.path``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx
from anthropic import AsyncAnthropic

if TYPE_CHECKING:
    from anthropic.types import MessageParam

#: Every browser ``Browser.stop`` was called on, by CDP URL.
stopped: list[str] = []


class ChatAnthropic:
    def __init__(self, *, model: str, api_key: str, base_url: str | None) -> None:
        self.model = model
        self.client = AsyncAnthropic(api_key=api_key, base_url=base_url, max_retries=0)


class Browser:
    def __init__(self, *, cdp_url: str) -> None:
        self.cdp_url = cdp_url

    async def stop(self) -> None:
        stopped.append(self.cdp_url)


@dataclass
class History:
    answer: str | None
    steps: int
    pages: list[str]

    def is_successful(self) -> bool | None:
        return self.answer is not None

    def final_result(self) -> str | None:
        return self.answer

    def number_of_steps(self) -> int:
        return self.steps

    def urls(self) -> list[str]:
        return self.pages

    def is_done(self) -> bool:
        return self.answer is not None


class Agent:
    def __init__(self, *, task: str, llm: ChatAnthropic, browser: Browser) -> None:
        self.task = task
        self.llm = llm
        self.browser = browser

    async def run(self, *, max_steps: int) -> History:
        async with httpx.AsyncClient() as http:
            targets = (await http.get(f"{self.browser.cdp_url}/json")).json()
        pages = [target["id"] for target in targets if target["type"] == "page"]
        messages: list[MessageParam] = [{"role": "user", "content": self.task}]
        for step in range(1, max_steps + 1):
            reply = await self.llm.client.messages.create(
                model=self.llm.model, max_tokens=1024, messages=messages
            )
            text = "".join(block.text for block in reply.content if block.type == "text")
            if text.startswith("done: "):
                return History(text.removeprefix("done: "), step, pages)
            messages += [
                {"role": "assistant", "content": text},
                {"role": "user", "content": "Continue."},
            ]
        return History(None, max_steps, pages)
