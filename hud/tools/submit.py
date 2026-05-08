from __future__ import annotations

import logging

from mcp.types import ContentBlock, TextContent

from .base import BaseTool

logger = logging.getLogger(__name__)


# Global submission storage
_SUBMISSION: str | None = None


def set_submission(value: str | None) -> None:
    global _SUBMISSION
    _SUBMISSION = value


def get_submission() -> str | None:
    return _SUBMISSION


class SubmitTool(BaseTool):
    """Lifecycle tool to submit the agent's final answer for evaluation.

    Accepts either a `response` string or a `messages` list and stores the
    submission as a plain string, accessible via `get_submission()`.
    Priority: The last text content in `messages` (if provided) overrides `response`.
    """

    name: str = "response"
    title: str = "Submit Tool"
    description: str = "Submit the agent's final response for later evaluation"

    def __init__(
        self,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
    ) -> None:
        super().__init__(
            name=name or self.name,
            title=title or self.title,
            description=description or self.description,
        )

    async def __call__(
        self, response: str | None = None, messages: list[ContentBlock] | None = None
    ) -> list[ContentBlock]:
        # 1) If messages provided, take the last text block
        # chosen: str | None = None

        # if messages:
        #     # Gather all text blocks
        #     text_blocks: list[str] = []
        #     for block in messages:
        #         try:
        #             if isinstance(block, TextContent):
        #                 text_blocks.append(str(block.text))
        #         except Exception:
        #             logger.debug("SubmitTool skipped non-text block: %s", block)
        #             continue
        #     if text_blocks:
        #         chosen = text_blocks[-1]

        # # 2) Otherwise use `response` as-is
        # if chosen is None and response is not None:
        #     chosen = response

        set_submission(response)

        # Echo back what we stored
        blocks: list[ContentBlock] = []
        if response:
            blocks.append(TextContent(text=response, type="text"))
        return blocks
