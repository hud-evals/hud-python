import base64
import io
from pathlib import Path
from typing import Any, Union

import torch
from transformers import ProcessorMixin, PreTrainedTokenizer
from PIL import Image
from transformers.utils.chat_template_utils import render_jinja_template

from hud.rl.types import ProcessedInputs

from hud.types import Trace


def preprocess_traces(traces: list[Trace], processor: Union[PreTrainedTokenizer, ProcessorMixin]) -> list[ProcessedInputs]:
    processed_inputs: list[ProcessedInputs] = []
    for trace in traces:
        if not trace.messages:
            continue

        conversation, images = prepare_conversation_history(trace.messages)

        chat_template_path = Path(__file__).parent / "chat_template.jinja"

        tokenizer = (
        processor.tokenizer if hasattr(processor, "tokenizer") else processor  # type: ignore
    )

        with open(chat_template_path) as f:
            chat_template = f.read()

        text_list, _ = render_jinja_template(
        conversations=[conversation],
        chat_template=chat_template,
        tools=trace.info["tool_spec"] if trace.info["tool_spec"] else None,
        **tokenizer.special_tokens_map,  # type: ignore
    )

        if hasattr(processor, "tokenizer"):
            inputs = processor(
            images=images if len(images) > 0 else None,
            text=text_list,
            return_offsets_mapping=False,
        )  # type: ignore
        else:
            inputs = processor(
            text=text_list,
            return_offsets_mapping=False,
        ) # type: ignore

        assistant_masks = build_assistant_masks(inputs["input_ids"], tokenizer)  # type: ignore
        mask_tensor = torch.tensor(assistant_masks, dtype=torch.long)

        inputs["assistant_mask"] = mask_tensor.bool()

        inputs.convert_to_tensors(tensor_type="pt")

        if "pixel_values" not in inputs:
            inputs["pixel_values"] = None
        if "image_grid_thw" not in inputs:
            inputs["image_grid_thw"] = None

        processed_inputs.append(inputs)  # type: ignore

    return processed_inputs


def prepare_conversation_history(
    conversation_history: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[Image.Image]]:
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
                    elif isinstance(data, (bytes, bytearray)):
                        images.append(Image.open(io.BytesIO(data)).convert("RGB"))
                    c = {"type": "image"}
            m["content"] = user_content
        sanitized_messages.append(m)
    return sanitized_messages, images


def b64_to_pil(b64_str: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


def build_assistant_masks(input_ids: list[list[int]], tokenizer: PreTrainedTokenizer) -> list[list[int]]:
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

                # Mark tokens until we hit <|im_end|>
                while i_tok < len(seq) and seq[i_tok] != id_im_end:
                    mask[i_tok] = 1
                    i_tok += 1
                
                # Include the <|im_end|> token
                mask[i_tok] = 1

            else:
                i_tok += 1

        assistant_masks.append(mask)

    return assistant_masks
