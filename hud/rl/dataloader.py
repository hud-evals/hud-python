from typing import TYPE_CHECKING, Any

import torch

from hud.rl.advantages import calculate_advantages
from hud.rl.buffer import Buffer
from hud.rl.logger import console
from hud.rl.preprocessor import preprocess_traces
from hud.rl.types import ProcessedInputs, TrainingSample

if TYPE_CHECKING:
    from hud.types import Trace
    from hud.rl.config import Config


class DataLoader:
    """Prepare batched training samples from rollouts stored in the buffer."""

    def __init__(
        self,
        buffer: Buffer,
        config: "Config",
        processor: Any | None = None,
        policy: Any | None = None,
    ) -> None:
        self.buffer = buffer
        self.config = config
        self.processor = processor
        self.policy = policy
        self.pad_token_id = self._resolve_pad_token_id(processor)

    def _resolve_pad_token_id(self, processor: Any | None) -> int:
        if processor is None:
            return 0
        tokenizer = getattr(processor, "tokenizer", processor)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            return 0
        return int(pad_token_id)

    def get_training_batch(self, traces: list["Trace"] | None = None) -> list[TrainingSample]:
        """Sample traces, preprocess them, and build model-ready training samples."""
        if traces is None:
            traces = self.buffer.sample_traces(self.buffer.batch_size)
        if not traces:
            raise ValueError("No traces available to construct a training batch.")

        rewards = torch.tensor(
            [0.0 if trace.isError else float(trace.reward) for trace in traces],
            dtype=torch.float32,
        )

        advantages = calculate_advantages(
            rewards=rewards,
            group_size=self.config.training.group_size,
            scale_rewards=self.config.training.scale_rewards,
            leave_one_out=self.config.training.leave_one_out,
        )

        processed_inputs = self._preprocess_traces(traces)
        if len(processed_inputs) != len(traces):
            raise ValueError(
                "Mismatch between number of traces and processed inputs; cannot build training batch."
            )

        samples: list[TrainingSample] = []
        for inputs, advantage in zip(processed_inputs, advantages, strict=True):
            processed = self._ensure_processed_inputs(inputs)
            samples.append(
                TrainingSample(
                    inputs=processed,
                    advantage=advantage.view(1),
                )
            )

        if self.config.training.accumulate_over_minibatches:
            return samples

        return self._batch_samples(samples)

    def _preprocess_traces(self, traces: list["Trace"]) -> list[ProcessedInputs]:
        if self.processor is None:
            return [self._empty_inputs() for _ in traces]
        return preprocess_traces(traces, self.processor)

    def _empty_inputs(self) -> ProcessedInputs:
        return {
            "input_ids": torch.zeros((1, 0), dtype=torch.long),
            "attention_mask": torch.zeros((1, 0), dtype=torch.long),
            "assistant_mask": torch.zeros((1, 0), dtype=torch.bool),
            "pixel_values": None,
            "image_grid_thw": None,
        }

    def _ensure_processed_inputs(self, inputs: Any) -> ProcessedInputs:
        data = dict(inputs)
        processed: ProcessedInputs = {
            "input_ids": self._ensure_2d_tensor(data.get("input_ids")),
            "attention_mask": self._ensure_2d_tensor(data.get("attention_mask")),
            "assistant_mask": self._ensure_2d_tensor(
                data.get("assistant_mask"), is_mask=True
            ),
            "pixel_values": data.get("pixel_values"),
            "image_grid_thw": data.get("image_grid_thw"),
        }
        return processed

    def _ensure_2d_tensor(self, tensor: torch.Tensor | None, *, is_mask: bool = False) -> torch.Tensor:
        if tensor is None:
            dtype = torch.bool if is_mask else torch.long
            return torch.zeros((1, 0), dtype=dtype)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() == 0:
            tensor = tensor.view(1, 1)
        elif tensor.dim() > 2:
            raise ValueError(f"Expected tensor with <=2 dims, got shape {tuple(tensor.shape)}")
        return tensor.bool() if is_mask else tensor

    def _batch_samples(self, samples: list[TrainingSample]) -> list[TrainingSample]:
        mini_batch_size = self.config.training.mini_batch_size
        if mini_batch_size <= 1:
            return samples

        batched_samples: list[TrainingSample] = []
        for i in range(0, len(samples), mini_batch_size):
            chunk = samples[i : i + mini_batch_size]
            if len(chunk) < mini_batch_size:
                console.warning_log(
                    f"Dropping incomplete minibatch of size {len(chunk)} (expected {mini_batch_size})."
                )
                break
            batched_samples.append(self._merge_chunk(chunk))

        return batched_samples

    def _merge_chunk(self, chunk: list[TrainingSample]) -> TrainingSample:
        seq_lengths = [
            self._strip_batch_dim(sample.inputs["input_ids"]).size(-1)
            for sample in chunk
        ]
        if not seq_lengths:
            raise ValueError("Cannot merge empty chunk of samples.")

        max_seq_len = max(seq_lengths)
        input_ids = torch.stack(
            [
                self._pad_sequence(
                    self._strip_batch_dim(sample.inputs["input_ids"]),
                    max_seq_len,
                    pad_value=self.pad_token_id,
                )
                for sample in chunk
            ],
            dim=0,
        )
        attention_mask = torch.stack(
            [
                self._pad_sequence(
                    self._strip_batch_dim(sample.inputs["attention_mask"]),
                    max_seq_len,
                    pad_value=0,
                )
                for sample in chunk
            ],
            dim=0,
        )

        mask_len = max(max_seq_len - 1, 0)
        assistant_mask = torch.stack(
            [
                self._pad_mask(
                    self._strip_batch_dim(sample.inputs["assistant_mask"]),
                    mask_len,
                )
                for sample in chunk
            ],
            dim=0,
        )

        merged_inputs: ProcessedInputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "assistant_mask": assistant_mask,
            "pixel_values": self._merge_optional_tensor(chunk, "pixel_values"),
            "image_grid_thw": self._merge_optional_tensor(chunk, "image_grid_thw"),
        }

        advantages = torch.cat([sample.advantage.view(-1) for sample in chunk], dim=0)
        return TrainingSample(inputs=merged_inputs, advantage=advantages)

    def _merge_optional_tensor(
        self, chunk: list[TrainingSample], key: str
    ) -> torch.Tensor | None:
        tensors: list[torch.Tensor] = []
        for sample in chunk:
            value = sample.inputs.get(key)
            if value is None:
                return None
            if value.dim() == 4:
                tensors.append(value.unsqueeze(0))
            else:
                tensors.append(value)
        if not tensors:
            return None
        return torch.cat(tensors, dim=0)

    def _strip_batch_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 2 and tensor.size(0) == 1:
            return tensor.squeeze(0)
        if tensor.dim() == 1:
            return tensor
        if tensor.dim() == 0:
            return tensor.view(1)
        raise ValueError(f"Unexpected tensor shape {tuple(tensor.shape)} while stripping batch dimension.")

    def _pad_sequence(
        self, tensor: torch.Tensor, target_len: int, pad_value: int
    ) -> torch.Tensor:
        current_len = tensor.size(-1)
        if current_len == target_len:
            return tensor
        padded = torch.full(
            (target_len,), pad_value, dtype=tensor.dtype, device=tensor.device
        )
        padded[:current_len] = tensor
        return padded

    def _pad_mask(self, tensor: torch.Tensor, target_len: int) -> torch.Tensor:
        current_len = tensor.size(-1)
        if current_len == target_len:
            return tensor.bool()
        padded = torch.zeros(target_len, dtype=torch.bool, device=tensor.device)
        truncated = tensor[:target_len].bool()
        padded[: truncated.size(-1)] = truncated
        return padded
