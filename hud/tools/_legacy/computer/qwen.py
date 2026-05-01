"""Legacy Qwen computer import path."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from hud.tools.computer import ComputerTool, computer_settings

if TYPE_CHECKING:
    from hud.tools.executors.base import BaseExecutor


class QwenComputerTool(ComputerTool):
    """Compatibility registration for Qwen computer use."""

    def __init__(
        self,
        executor: BaseExecutor | None = None,
        platform_type: Literal["auto", "xdo", "pyautogui"] = "auto",
        display_num: int | None = None,
        width: int = computer_settings.QWEN_COMPUTER_WIDTH,
        height: int = computer_settings.QWEN_COMPUTER_HEIGHT,
        rescale_images: bool = computer_settings.QWEN_RESCALE_IMAGES,
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
            name=name or "qwen_computer",
            title=title or "Computer Control",
            description=description or "Control computer with mouse, keyboard, and screenshots",
            **kwargs,
        )

__all__ = ["QwenComputerTool"]
