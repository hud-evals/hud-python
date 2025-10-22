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
        super().info_log(self._format_msg(msg), stderr=stderr)

    def warning(self, msg: str, stderr: bool = True) -> None:
        super().warning(self._format_msg(msg), stderr=stderr)

    def warning_log(self, msg: str, stderr: bool = True) -> None:
        super().warning_log(self._format_msg(msg), stderr=stderr)

    def error(self, msg: str, stderr: bool = True) -> None:
        super().error(self._format_msg(msg), stderr=stderr)

    def error_log(self, msg: str, stderr: bool = True) -> None:
        super().error_log(self._format_msg(msg), stderr=stderr)

    def debug_log(self, msg: str, stderr: bool = True) -> None:
        super().debug_log(self._format_msg(msg), stderr=stderr)


def setup_logger() -> RankAwareHUDConsole:
    """Single shared instance that automatically adds rank info."""
    logger = logging.getLogger("hud.rl")
    return RankAwareHUDConsole(logger=logger)


def configure_logging(verbosity: int = 0) -> None:
    """Configure logging level based on verbose flag.
    
    Args:
        verbosity: If 0, set to WARNING level. If 1, set to INFO level. If 2, set to DEBUG level.
    """
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity == 2:
        level = logging.DEBUG
    logging.basicConfig(level=level)


console = setup_logger()
