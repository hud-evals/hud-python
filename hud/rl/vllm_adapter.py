"""vLLM adapter management for LoRA hot-swapping."""

from __future__ import annotations

import json
import logging

import requests

from hud.utils.hud_console import HUDConsole

hud_console = HUDConsole(logging.getLogger(__name__))


class VLLMAdapter:
    """Manages LoRA adapter loading/unloading in vLLM."""

    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.current_adapter = None

    def load_adapter(self, adapter_name: str, adapter_path: str, timeout: int = 30) -> bool:
        """
        Hot-load a LoRA adapter to vLLM.

        Args:
            adapter_name: Name to register the adapter as
            adapter_path: Path to the adapter checkpoint
            timeout: Request timeout in seconds

        Returns:
            True if successful, False otherwise
        """
        url = f"{self.base_url}/load_lora_adapter"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"lora_name": adapter_name, "lora_path": adapter_path}
        # Implement exponential backoff for retrying the adapter load request.
        max_retries = 5
        backoff_factor = 2
        delay = 1  # initial delay in seconds

        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(
                    url, headers=headers, data=json.dumps(payload), timeout=timeout
                )
                response.raise_for_status()

                self.current_adapter = adapter_name
                hud_console.info(f"[VLLMAdapter] Loaded adapter: {adapter_name}")
                return True

            except requests.exceptions.RequestException as e:
                if attempt == max_retries:
                    hud_console.error(
                        f"[VLLMAdapter] Failed to load adapter {adapter_name} after {attempt} attempts: {e}"  # noqa: E501
                    )
                    return False
                else:
                    hud_console.warning(
                        f"[VLLMAdapter] Load adapter {adapter_name} failed (attempt {attempt}/{max_retries}): {e}. Retrying in {delay} seconds...",  # noqa: E501
                    )
                    import time

                    time.sleep(delay)
                    delay *= backoff_factor

        return False

    def unload_adapter(self, adapter_name: str) -> bool:
        """
        Unload a LoRA adapter from vLLM.

        Args:
            adapter_name: Name of the adapter to unload

        Returns:
            True if successful, False otherwise
        """
        url = f"{self.base_url}/unload_lora_adapter"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"lora_name": adapter_name}

        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
            response.raise_for_status()

            if self.current_adapter == adapter_name:
                self.current_adapter = None

            hud_console.info(f"[VLLMAdapter] Unloaded adapter: {adapter_name}")
            return True

        except requests.exceptions.RequestException as e:
            hud_console.error(f"[VLLMAdapter] Failed to unload adapter {adapter_name}: {e}")
            return False

    def list_adapters(self) -> list | None:
        """
        List all loaded LoRA adapters in vLLM.

        Returns:
            List of adapter names, or None if failed
        """
        url = f"{self.base_url}/list_lora_adapters"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json().get("adapters", [])

        except requests.exceptions.RequestException as e:
            hud_console.error(f"[VLLMAdapter] Failed to list adapters: {e}")
            return None

    def get_current(self) -> str | None:
        """Get the name of the currently loaded adapter."""
        return self.current_adapter


# Convenience function for standalone use
def hotload_lora(
    adapter_name: str,
    adapter_path: str,
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "token-abc123",
) -> bool:
    """
    Quick function to hot-load a LoRA adapter.

    Args:
        adapter_name: Name for the adapter
        adapter_path: Path to adapter checkpoint
        base_url: vLLM server URL
        api_key: API key for vLLM

    Returns:
        True if successful
    """
    adapter = VLLMAdapter(base_url, api_key)
    return adapter.load_adapter(adapter_name, adapter_path)
