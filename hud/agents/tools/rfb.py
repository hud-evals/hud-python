"""RFBTool: capability base for tools driven by an ``RFBClient``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from hud.agents.computer.screen import Screen
from hud.agents.tools.base import AgentTool, AgentToolSpec
from hud.capabilities.rfb import RFBClient, ScreenshotEncoding

if TYPE_CHECKING:
    from hud.agents.types import ToolAgentConfig


class RFBTool(AgentTool[RFBClient], Screen):
    """Capability base: tool driven by an ``RFBClient`` (VNC/RFB)."""

    client_type = RFBClient

    def __init__(
        self,
        *,
        spec: AgentToolSpec,
        client: RFBClient,
        screenshot_encoding: ScreenshotEncoding,
    ) -> None:
        AgentTool.__init__(self, spec=spec, client=client)
        Screen.__init__(self, client, screenshot_encoding)

    @classmethod
    def from_config(
        cls, *, spec: AgentToolSpec, client: RFBClient, config: ToolAgentConfig
    ) -> Self:
        return cls(spec=spec, client=client, screenshot_encoding=config.screenshot_encoding)
