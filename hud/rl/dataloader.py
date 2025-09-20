from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from hud.rl.logger import console
from hud.rl.types import TrainingSample
from hud.rl.advantages import calculate_advantages
from hud.rl.preprocessor import prepare_inputs_for_samples

if TYPE_CHECKING:
    from hud.types import Trace

    from hud.rl.buffer import Buffer
    from hud.rl.config import Config


class DataLoader:
    """DataLoader that wraps buffers and handles data preprocessing."""

    def __init__(
        self,
        buffer: Buffer,
        config: Config,
        processor: Any | None = None,
        policy: Any | None = None,
    ) -> None:
        """Initialize DataLoader.

        Args:
            buffer: Buffer instance for sampling tasks and traces
            config: Training configuration
            processor: Model processor for tokenization
            policy: Policy model for computing logprobs
        """
        self.buffer = buffer
        self.config = config
        self.processor = processor
        self.policy = policy

    def get_tasks(self, n: int) -> list[Any]:
        """Get tasks from the buffer for actor execution."""
        return self.buffer.sample_tasks(n)

    def get_training_batch(self) -> list[TrainingSample]:
        """Get a preprocessed training batch.

        Returns:
            List of TrainingSample objects ready for training
        """
        # Sample traces from buffer
        traces = self.buffer.sample_traces()

        # Calculate advantages
        samples = calculate_advantages(traces, self.config)

        # Prepare inputs if processor is available
        if self.processor is not None:
            samples = prepare_inputs_for_samples(samples, self.processor)

        # Batch samples if needed
        if self.config.training.accumulate_over_minibatches:
            return samples
        else:
            # Batch samples for efficient forward pass
            return self._batch_samples(samples)



    def _batch_samples(self, samples: list[TrainingSample]) -> list[TrainingSample]:
        """Batch samples for efficient processing.

        Args:
            samples: List of TrainingSample objects

        Returns:
            List of batched TrainingSample objects
        """
        mini_batch_size = self.config.training.mini_batch_size
        batched_samples = []

        for i in range(0, len(samples), mini_batch_size):
            mini_batch = samples[i : i + mini_batch_size]
            if len(mini_batch) > 1:
                # Batch multiple samples together
                batched = self._batch_training_samples(mini_batch)
                batched_samples.extend(batched)
            else:
                batched_samples.extend(mini_batch)

        return batched_samples

    def _batch_training_samples(self, samples: list[TrainingSample]) -> list[TrainingSample]:
        """Create batched model inputs from a list of TrainingSample.

        Pads token sequences to the maximum length in the list and zero-pads
        images to the maximum H/W when present.
        """
        if not samples:
            console.warning_log("No samples to batch.")
            return []

        # Remove samples with zero advantage
        for s in samples:
            if (
                "assistant_mask" not in s.inputs
                or s.inputs["assistant_mask"].sum() == 0
                or s.advantage == 0.0
            ) and len(samples) > 1:
                console.info_log("Removing sample with zero advantage.")
                samples.remove(s)

        if len(samples) == 1:
            return samples

        new_samples = [TrainingSample()]

        input_keys_to_expand = ["input_ids", "attention_mask", "assistant_mask"]
        input_keys_to_cat = ["pixel_values", "image_grid_thw"]
        updated_inputs: dict[str, list[torch.Tensor]] = {
            k: [] for k in input_keys_to_expand + input_keys_to_cat
        }

        # Sanity check dimensions
        for s in samples:
            for k in input_keys_to_expand + input_keys_to_cat:
                val = s.inputs.get(k)
                if val is not None:
                    if k in input_keys_to_expand:
                        if val.dim() == 2 and val.size(0) == 1:
                            val = val[0]
                        elif val.dim() != 1:
                            raise ValueError(f"{k} has unexpected dimensions: {val.shape}")
                    updated_inputs[k].append(val)

        # Pad 1D sequences to max length
        max_len = max(t.size(-1) for t in updated_inputs["input_ids"])

        def pad_1d(x: torch.Tensor, pad_to: int, pad_value: int) -> torch.Tensor:
            pad = pad_to - x.size(-1)
            return F.pad(x, (0, pad), value=pad_value) if pad > 0 else x

        stacked_inputs: dict[str, torch.Tensor] = {}
        # These are 1D sequences that need padding
        for k in input_keys_to_expand:
            if updated_inputs[k]:
                # assistant_mask is T-1, others are T
                if k == "assistant_mask":
                    stacked_inputs[k] = torch.stack(
                        [pad_1d(x, max_len - 1, 0) for x in updated_inputs[k]], dim=0
                    )
                else:
                    stacked_inputs[k] = torch.stack(
                        [pad_1d(x, max_len, 0) for x in updated_inputs[k]], dim=0
                    )

        for k in input_keys_to_cat:
            if updated_inputs[k]:
                # pixel_values and image_grid_thw are concatenated across all images
                stacked_inputs[k] = torch.cat(updated_inputs[k], dim=0)

        new_samples[0].inputs = stacked_inputs

        # Pad logprobs to max length before stacking
        def pad_logprobs(logprobs: torch.Tensor | None, max_len: int) -> torch.Tensor:
            # Always work with 1D tensor, squeeze batch dim if present
            if logprobs is None:
                return torch.tensor([float("-inf")], dtype=torch.float32)
            if logprobs.dim() == 2 and logprobs.size(0) == 1:
                logprobs = logprobs.squeeze(0)
            elif logprobs.dim() != 1:
                raise ValueError(
                    f"Expected logprobs to have 1 or 2 dimensions, got {logprobs.dim()} with shape {logprobs.shape}"
                )

            # Now logprobs is [seq_len]
            seq_len = logprobs.size(0) if logprobs is not None else 0
            if seq_len < max_len:
                pad_size = max_len - seq_len
                # Pad with -inf (log of 0 probability) along sequence dimension
                return F.pad(logprobs, (0, pad_size), value=float("-inf"))
            return logprobs

        # Stack padded logprobs (these are T-1 length)
        old_logprobs_list = [pad_logprobs(s.old_logprobs, max_len - 1) for s in samples]
        ref_logprobs_list = [pad_logprobs(s.ref_logprobs, max_len - 1) for s in samples]

        new_samples[0].old_logprobs = torch.stack(old_logprobs_list, dim=0)
        new_samples[0].ref_logprobs = torch.stack(ref_logprobs_list, dim=0)

        # Stack advantages, checking for None values
        advantages = [s.advantage for s in samples]
        if any(adv is None for adv in advantages):
            raise ValueError(
                "Some samples have None advantages. Make sure advantages are computed before batching."
            )
        new_samples[0].advantage = torch.stack(advantages, dim=0)  # type: ignore

        return new_samples

    def add_traces(self, traces: list[Trace]) -> None:
        """Add completed traces to the buffer.

        Args:
            traces: List of completed traces from actor
        """
        self.buffer.add_traces(traces)

    def update_buffer(self, **kwargs) -> None:
        """Update buffer state (e.g., for curriculum learning).

        Args:
            **kwargs: Arguments to pass to buffer.update()
        """
        self.buffer.update(**kwargs)
