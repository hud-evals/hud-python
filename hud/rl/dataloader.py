import base64
import io
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers.utils.chat_template_utils import render_jinja_template

from hud.rl.logger import console

from hud.rl.types import TrainingSample

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

        # Preprocess advantages
        samples = self._preprocess_advantages(traces)

        # Prepare inputs if processor is available
        if self.processor is not None:
            samples = self._prepare_inputs_for_samples(samples)

        # Batch samples if needed
        if self.config.training.accumulate_over_minibatches:
            return samples
        else:
            # Batch samples for efficient forward pass
            return self._batch_samples(samples)

    def _preprocess_advantages(self, traces: list[Trace]) -> list[TrainingSample]:
        """Preprocess a group of traces to compute advantages.

        Args:
            traces: List of traces to process

        Returns:
            List of TrainingSample objects with computed advantages
        """
        group_size = self.config.training.group_size
        batch_level = self.config.training.batch_level

        if batch_level == "group":
            groups = [traces[i : i + group_size] for i in range(0, len(traces), group_size)]
        elif batch_level == "batch":
            groups = [traces]
        else:
            raise ValueError(f"Invalid batch level: {batch_level}")

        all_samples = []
        for i, group in enumerate(groups):
            rewards = np.array([trace.reward for trace in group])
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)

            # Calculate advantages
            samples = [TrainingSample(**trace.model_dump()) for trace in group]
            for sample, reward in zip(samples, rewards, strict=True):
                if sample.isError:
                    sample.advantage = torch.tensor([0.0])
                    continue

                # No std (non-baseline GRPO)
                if self.config.training.no_std:
                    advantage_value = reward - mean_reward
                else:
                    # Avoid division by zero
                    if std_reward < 1e-6:
                        advantage_value = 0.0
                    else:
                        advantage_value = (reward - mean_reward) / std_reward

                # Leave one out RLOO/LOOP
                if self.config.training.leave_one_out:
                    advantage_value = advantage_value * len(group) / (len(group) - 1)

                sample.advantage = torch.tensor([advantage_value])

            console.info_log(
                f"Advantages for group {i} [{mean_reward:.4f} ± {std_reward:.4f}]: "
                f"{[round(sample.advantage.item(), 4) for sample in samples if sample.advantage is not None]}"
            )

            all_samples.extend(samples)

        return all_samples

    def _prepare_inputs_for_samples(self, samples: list[TrainingSample]) -> list[TrainingSample]:
        """Prepare inputs for each sample using the processor.

        Args:
            samples: List of TrainingSample objects

        Returns:
            List of TrainingSample objects with tokenized inputs
        """
        for sample in samples:
            if sample.messages:
                inputs = self._prepare_inputs(sample)
                if not inputs or "input_ids" not in inputs:
                    # Create dummy inputs for invalid samples
                    console.warning_log("Sample has invalid inputs, using dummy values")
                    inputs = {
                        "input_ids": torch.zeros(1, 2, dtype=torch.long),
                        "attention_mask": torch.ones(1, 2, dtype=torch.long),
                        "assistant_mask": torch.zeros(1, 1, dtype=torch.bool),
                    }
                elif "assistant_mask" not in inputs:
                    console.warning_log("Sample missing assistant_mask, creating zero mask")
                    seq_len = inputs["input_ids"].shape[-1]
                    inputs["assistant_mask"] = torch.zeros(
                        inputs["input_ids"].shape[0], seq_len - 1, dtype=torch.bool
                    )
                sample.inputs = inputs
        return samples

    def _prepare_inputs(self, trace: Trace) -> dict[str, torch.Tensor]:
        """Prepare inputs from a trace.

        Args:
            trace: Trace to process

        Returns:
            Inputs for the model
        """
        if len(trace.messages) == 0:
            return {}

        # Get images for current turn
        conversation, images = self._prepare_conversation_history(trace.messages)

        # Get absolute path to chat template
        chat_template_path = Path(__file__).parent / "chat_template.jinja"

        # For VL models, processor has a tokenizer attribute; for text models, processor IS tokenizer
        tokenizer = (
            self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor
        )

        # Load chat template
        with open(chat_template_path) as f:
            chat_template = f.read()

        text_list, _ = render_jinja_template(
            conversations=[conversation],
            chat_template=chat_template,
            tools=trace.info["tool_spec"] if trace.info["tool_spec"] else None,
            return_assistant_tokens_mask=True,
            **tokenizer.special_tokens_map,
        )

        # For text models, don't pass images parameter
        if hasattr(self.processor, "tokenizer"):
            # VL model - processor accepts images
            inputs = self.processor(
                images=images if len(images) > 0 else None,
                text=text_list,
                return_offsets_mapping=False,
            )
        else:
            # Text model - processor is tokenizer, doesn't accept images
            inputs = self.processor(
                text=text_list,
                return_offsets_mapping=False,
            )

        assistant_masks = self._build_assistant_masks(inputs["input_ids"], tokenizer)
        mask_tensor = torch.tensor(assistant_masks, dtype=torch.long)

        # Ensure mask_tensor is 2D before slicing
        if mask_tensor.dim() == 1:
            mask_tensor = mask_tensor.unsqueeze(0)

        # Slice to align with targets [B, T-1]
        inputs["assistant_mask"] = mask_tensor[:, 1:].bool()

        inputs.convert_to_tensors(tensor_type="pt")

        return inputs

    def _prepare_conversation_history(
        self, conversation_history: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[Image.Image]]:
        """Sanitize conversation history to avoid vLLM errors."""
        sanitized_messages = []
        images = []
        for m in conversation_history:
            if "tool_calls" in m:
                m = {
                    "role": m["role"],
                    "content": m.get("content", ""),
                    "tool_calls": [
                        tc.model_dump() if not isinstance(tc, dict) else tc
                        for tc in m.get("tool_calls", [])
                    ],
                }
            elif m.get("role") == "user":
                user_content = m.get("content", [])
                for c in user_content:
                    if isinstance(c, dict) and c.get("type") == "image_url":
                        image_url = c.get("image_url", {})
                        url = image_url.get("url", "")
                        if url.startswith("data:image"):
                            data = url.split(",", 1)[1] if "," in url else url
                            images.append(self._b64_to_pil(data))
                        elif isinstance(data, bytes | bytearray):
                            images.append(Image.open(io.BytesIO(data)).convert("RGB"))
                        c = {"type": "image"}
                m["content"] = user_content
            sanitized_messages.append(m)
        return sanitized_messages, images

    def _b64_to_pil(self, b64_str: str) -> Image.Image:
        """Convert base64 string to PIL Image."""
        return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")

    def _build_assistant_masks(self, input_ids: list[list[int]], tokenizer: Any) -> list[list[int]]:
        """Build assistant masks from token IDs by finding assistant turns.

        Args:
            input_ids: List of token sequences
            tokenizer: Tokenizer to decode tokens and get special token IDs

        Returns:
            List of binary masks indicating assistant tokens
        """
        id_im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
        id_im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        id_assistant = tokenizer.convert_tokens_to_ids("assistant")

        assistant_masks: list[list[int]] = []

        for seq in input_ids:
            mask = [0] * len(seq)
            i_tok = 0

            while i_tok < len(seq):
                # Detect start of assistant turn
                if (
                    seq[i_tok] == id_im_start
                    and i_tok + 1 < len(seq)
                    and seq[i_tok + 1] == id_assistant
                ):
                    # Skip '<|im_start|>', 'assistant' and possible newline token
                    i_tok += 2
                    # Check for newline after 'assistant'
                    if i_tok < len(seq) and tokenizer.decode([seq[i_tok]]) == "\n":
                        i_tok += 1

                    # Skip leading spaces after assistant\n
                    while i_tok < len(seq) and tokenizer.decode([seq[i_tok]]).strip() == "":
                        i_tok += 1

                    assistant_content_start = i_tok

                    # Mark tokens until we hit <|im_end|>
                    content_end = i_tok
                    while i_tok < len(seq) and seq[i_tok] != id_im_end:
                        content_end = i_tok + 1  # Track last non-<|im_end|> position
                        mask[i_tok] = 1
                        i_tok += 1

                    # Remove trailing spaces from the mask
                    while content_end > assistant_content_start:
                        if (
                            mask[content_end - 1] == 1
                            and tokenizer.decode([seq[content_end - 1]]).strip() == ""
                        ):
                            mask[content_end - 1] = 0
                            content_end -= 1
                        else:
                            break

                    # Skip the <|im_end|> token
                    i_tok += 1
                else:
                    i_tok += 1

            assistant_masks.append(mask)

        return assistant_masks

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
