"""Kernel browser provider implementation (stub)."""

from .base import BrowserProvider


class KernelProvider(BrowserProvider):
    """Kernel browser-as-a-service platform - placeholder implementation."""

    async def launch(self, **kwargs) -> str:
        raise NotImplementedError("Kernel provider not yet implemented")

    def close(self) -> None:
        raise NotImplementedError("Kernel provider not yet implemented")
