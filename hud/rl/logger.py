import logging
import os

from hud.utils.hud_console import HUDConsole


class RankAwareHUDConsole(HUDConsole):
    """HUDConsole that automatically prepends rank info to messages."""

    def _format_msg(self, msg: str) -> str:
        """Add rank prefix if available."""
        rank = os.environ.get("RANK")
        if rank is not None:
            return f"[rank {rank}] {msg}"
        return msg

    def info(self, msg: str) -> None:
        super().info(self._format_msg(msg))

    def info_log(self, msg: str) -> None:
        super().info_log(self._format_msg(msg))

    def warning(self, msg: str) -> None:
        super().warning(self._format_msg(msg))

    def warning_log(self, msg: str) -> None:
        super().warning_log(self._format_msg(msg))

    def error(self, msg: str) -> None:
        super().error(self._format_msg(msg))


# Single shared instance that automatically adds rank info
logger = logging.getLogger("hud.rl")
console = RankAwareHUDConsole(logger=logger)
