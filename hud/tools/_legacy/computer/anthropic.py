"""Legacy Anthropic computer import path."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from hud.tools.computer import ComputerTool

if TYPE_CHECKING:
    from hud.tools.executors.base import BaseExecutor


class AnthropicComputerTool(ComputerTool):
    """Compatibility registration for Claude computer use."""

    def __init__(
        self,
        executor: BaseExecutor | None = None,
        platform_type: Literal["auto", "xdo", "pyautogui"] = "auto",
        display_num: int | None = None,
        width: int | None = None,
        height: int | None = None,
        rescale_images: bool = False,
        screenshot_quality: int | None = None,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            executor=executor,
            platform_type=platform_type,
            display_num=display_num,
            width=width,
            height=height,
            rescale_images=rescale_images,
            name=name or "anthropic_computer",
            title=title or "Computer Control",
            description=description or "Control computer with mouse, keyboard, and screenshots",
            **kwargs,
        )
        self.screenshot_quality = screenshot_quality

__all__ = ["AnthropicComputerTool"]
