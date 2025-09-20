import os
from datetime import datetime
from pathlib import Path
from typing import Any

from hud.rl.distributed import get_global_rank, is_main_process
from hud.rl.logger import console


class CheckpointManager:
    """Manages checkpoint saving, loading, and path generation for training."""

    def __init__(self, out_dir: str, adapter_prefix: str = "checkpoint"):
        self.out_dir = out_dir
        self.adapter_prefix = adapter_prefix

        # Ensure output directory exists
        os.makedirs(out_dir, exist_ok=True)

    def save(self, policy: Any, path: str | None = None) -> str:
        """Save the current policy checkpoint (only on rank 0).

        Args:
            policy: The policy model to save
            path: Optional custom path. If None, generates timestamped path

        Returns:
            Path where checkpoint was saved
        """
        if path is None:
            path, _ = self.create_timestamped_path()

        if is_main_process():
            os.makedirs(path, exist_ok=True)
            # Unwrap DDP model if needed
            model_to_save = policy.module if hasattr(policy, "module") else policy
            model_to_save.save_pretrained(path)
            console.info_log(f"Saved checkpoint to {path}")

        return path

    def load(self, policy: Any, path: str) -> None:
        """Load a policy checkpoint.

        Args:
            policy: The policy model to load into
            path: Directory path to load the checkpoint from

        Note:
            This is a placeholder implementation. Full implementation would depend
            on PEFT version and require reloading LoRA weights.
        """
        console.info_log(f"Loading checkpoint from {path}")
        # TODO: Implement full checkpoint loading logic
        # This would need to:
        # 1. Load the base model weights
        # 2. Load the LoRA adapter weights
        # 3. Reconstruct the PEFT model
        # Implementation depends on PEFT version

    def create_timestamped_path(self) -> tuple[str, str]:
        """Create a checkpoint path with timestamp and adapter name.

        Returns:
            Tuple of (checkpoint_path, adapter_name)
        """
        now = datetime.now()
        rank = get_global_rank()
        checkpoint_id = now.strftime("%Y%m%d_%H%M%S") + f"-{rank}"
        adapter_name = f"{self.adapter_prefix}-{checkpoint_id}"
        checkpoint_path = str(Path(self.out_dir) / adapter_name)
        return checkpoint_path, adapter_name
