import shutil
import warnings
from pathlib import Path

import torch
from torch import nn, Tensor
from torch.distributed.checkpoint.state_dict import _get_fqns as get_fqns
from torch.distributed.tensor import DTensor

from hud.rl.utils import is_main_process
from hud.rl.logger import console


class CheckpointManager:
    """Manages checkpoint saving and cleanup for training."""

    def __init__(self, output_dir: str | Path, save_last_n: int = 1):
        self.output_dir = Path(output_dir)
        self.save_last_n = save_last_n
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._is_master = is_main_process()
        
        self._saved_steps: list[int] = []
        
        self._best_metric: float | None = None
        self._best_step: int | None = None

    def _get_checkpoint_path(self, step: int) -> Path:
        return self.output_dir / f"step_{step:05d}" / "checkpoints"

    def _gather_weights(self, model: nn.Module) -> dict[str, Tensor]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="torch.distributed")
            warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed.*")

            cpu_state = {}
            for key, value in model.state_dict().items():
                if isinstance(value, DTensor):
                    # Gather full tensor from all ranks
                    value = value.full_tensor()

                if self._is_master:
                    # Get fully qualified name for HF compatibility
                    fqn = get_fqns(model, key)
                    assert len(fqn) == 1
                    fqn = next(iter(fqn))
                    cpu_state[fqn] = value.to("cpu", non_blocking=False)

            torch.distributed.barrier()

        return cpu_state

    def save(
        self,
        model: nn.Module,
        step: int,
    ) -> Path | None:
        checkpoint_path = self._get_checkpoint_path(step)
        
        # Gather weights on master rank
        cpu_state = self._gather_weights(model)
        
        if self._is_master:
            console.info(f"Saving checkpoint to {checkpoint_path}")
            
            temp_checkpoint_path = checkpoint_path.parent / f"{checkpoint_path.name}.tmp"
            temp_checkpoint_path.mkdir(parents=True, exist_ok=True)
            
            model_path = temp_checkpoint_path / "model.safetensors"
            torch.save(cpu_state, model_path)
            
            if hasattr(model, 'config'):
                model.config.save_pretrained(temp_checkpoint_path) # type: ignore
            if hasattr(model, 'generation_config') and model.generation_config:
                model.generation_config.save_pretrained(temp_checkpoint_path)  # type: ignore
            
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
            temp_checkpoint_path.rename(checkpoint_path)
            
            console.info(f"Checkpoint saved to {checkpoint_path}")
            
            self._saved_steps.append(step)
            self._cleanup_old_checkpoints()
            
            return checkpoint_path
        
        return None

    def _cleanup_old_checkpoints(self) -> None:
        if self.save_last_n == -1:
            return
        self._saved_steps.sort()
        
        steps_to_delete = self._saved_steps[:-self.save_last_n]
        
        for step in steps_to_delete:
            if self._best_step is not None and step == self._best_step:
                continue
            
            step_dir = self.output_dir / f"step_{step:05d}"
            if step_dir.exists():
                console.info_log(f"Removing old step directory: {step_dir}")
                shutil.rmtree(step_dir, ignore_errors=True)
        
        self._saved_steps = [s for s in self._saved_steps if s not in steps_to_delete or s == self._best_step]

    def maybe_save_best(
        self,
        model: nn.Module,
        step: int,
        metric: float,
        reverse: bool = True,
    ) -> Path | None:
        """
        Optionally save checkpoint if it's the best so far based on a metric. reverse: Whether higher metric values are better
        """
        is_best = False
        
        if self._best_metric is None:
            is_best = True
        elif not reverse and metric > self._best_metric:
            is_best = True
        elif reverse and metric < self._best_metric:
            is_best = True
        
        if is_best and self._is_master:
            self._best_metric = metric
            self._best_step = step
            
            best_path = self.output_dir / "best"
            if best_path.exists():
                shutil.rmtree(best_path, ignore_errors=True)
            
            checkpoint_path = self._get_checkpoint_path(step)
            if checkpoint_path.exists():
                shutil.copytree(checkpoint_path, best_path)
                console.info(f"Saved best checkpoint (step {step}, metric={metric:.4f}) to {best_path}")
                return best_path
        
        return None

    def load(self, model: nn.Module, checkpoint_path: str | Path) -> None:
        checkpoint_path = Path(checkpoint_path)
        model_path = checkpoint_path / "model.safetensors"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")
        
        console.info(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(model_path, weights_only=False)
        model.load_state_dict(state_dict, strict=True)
        console.info("Checkpoint loaded successfully")
