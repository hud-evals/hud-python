import logging
import os

from hud.utils.hud_console import HUDConsole


class RankAwareHUDConsole(HUDConsole):
    """HUDConsole that automatically prepends rank info to messages."""

    def _format_msg(self, msg: str) -> str:
        """Add rank prefix if available."""
        rank = os.environ.get("RANK")
        if rank is not None:
            return f"\\[rank {rank}] {msg}"
        return msg

    def info(self, msg: str, stderr: bool = True) -> None:
        super().info(self._format_msg(msg), stderr=stderr)

    def info_log(self, msg: str, stderr: bool = True) -> None:
        super().info_log(msg, stderr=stderr)

    def warning(self, msg: str, stderr: bool = True) -> None:
        super().warning(self._format_msg(msg), stderr=stderr)

    def warning_log(self, msg: str, stderr: bool = True) -> None:
        super().warning_log(msg, stderr=stderr)

    def error(self, msg: str, stderr: bool = True) -> None:
        super().error(self._format_msg(msg), stderr=stderr)


def setup_logger() -> RankAwareHUDConsole:
# Single shared instance that automatically adds rank info
    logger = logging.getLogger("hud.rl")
    return RankAwareHUDConsole(logger=logger)

console = setup_logger()
