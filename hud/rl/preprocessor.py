"""Input preprocessing utilities for GRPO training."""

import base64
import io
from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch
from PIL import Image
from transformers.utils.chat_template_utils import render_jinja_template

from hud.rl.logger import console

if TYPE_CHECKING:
    from hud.types import Trace
    from hud.rl.types import TrainingSample


def prepare_inputs_for_samples(samples: list[TrainingSample], processor: Any) -> list[TrainingSample]:
    """Prepare inputs for each sample using the processor.

    Args:
        samples: List of TrainingSample objects
        processor: Model processor for tokenization

    Returns:
        List of TrainingSample objects with tokenized inputs
    """
    for sample in samples:
        if sample.messages:
            inputs = prepare_inputs(sample, processor)
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


def prepare_inputs(trace: Trace, processor: Any) -> dict[str, torch.Tensor]:
    """Prepare inputs from a trace.

    Args:
        trace: Trace to process
        processor: Model processor for tokenization

    Returns:
        Inputs for the model
    """
    if len(trace.messages) == 0:
        return {}

    # Get images for current turn
    conversation, images = prepare_conversation_history(trace.messages)

    # Get absolute path to chat template
    chat_template_path = Path(__file__).parent / "chat_template.jinja"

    # For VL models, processor has a tokenizer attribute; for text models, processor IS tokenizer
    tokenizer = (
        processor.tokenizer if hasattr(processor, "tokenizer") else processor
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
    if hasattr(processor, "tokenizer"):
        # VL model - processor accepts images
        inputs = processor(
            images=images if len(images) > 0 else None,
            text=text_list,
            return_offsets_mapping=False,
        )
    else:
        # Text model - processor is tokenizer, doesn't accept images
        inputs = processor(
            text=text_list,
            return_offsets_mapping=False,
        )

    assistant_masks = build_assistant_masks(inputs["input_ids"], tokenizer)
    mask_tensor = torch.tensor(assistant_masks, dtype=torch.long)

    # Ensure mask_tensor is 2D before slicing
    if mask_tensor.dim() == 1:
        mask_tensor = mask_tensor.unsqueeze(0)

    # Slice to align with targets [B, T-1]
    inputs["assistant_mask"] = mask_tensor[:, 1:].bool()

    inputs.convert_to_tensors(tensor_type="pt")

    return inputs


def prepare_conversation_history(
    conversation_history: list[dict[str, Any]]
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
                        images.append(b64_to_pil(data))
                    elif isinstance(data, bytes | bytearray):
                        images.append(Image.open(io.BytesIO(data)).convert("RGB"))
                    c = {"type": "image"}
            m["content"] = user_content
        sanitized_messages.append(m)
    return sanitized_messages, images


def b64_to_pil(b64_str: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


def build_assistant_masks(input_ids: list[list[int]], tokenizer: Any) -> list[list[int]]:
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
