import os
from datetime import datetime
from pathlib import Path
from typing import Any

class CheckpointManager:
    """Manages checkpoint saving, loading, and path generation for training."""

    def __init__(self, out_dir: str, checkpoint_prefix: str = "checkpoint"):
        self.out_dir = out_dir
        self.checkpoint_prefix = checkpoint_prefix

        # Ensure output directory exists
        os.makedirs(out_dir, exist_ok=True)

    def save(self, model: Any, path: str | None = None) -> None:
        # TODO
        pass

    def load(self, model: Any, path: str) -> None:
        # TODO
        pass

    def create_timestamped_path(self) -> tuple[str, str]:
        now = datetime.now()
        checkpoint_id = now.strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"{self.checkpoint_prefix}-{checkpoint_id}"
        checkpoint_path = str(Path(self.out_dir) / checkpoint_name)
        return checkpoint_path, checkpoint_name
