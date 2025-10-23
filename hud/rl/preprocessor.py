import base64
import io
import json
from dataclasses import dataclass
from typing import Any, Iterator, Union

import torch
from PIL import Image
from transformers import ProcessorMixin

from hud.rl.types import ProcessedInputs
from hud.types import Trace


@dataclass
class AgentTurnTokens:
    prompt_token_ids: list[int]
    prompt_logprobs: list[float]
    completion_token_ids: list[int]
    completion_logprobs: list[float]


def preprocess_traces(
    traces: list[Trace],
    processor: Union[ProcessorMixin, Any],
) -> list[tuple[ProcessedInputs, torch.Tensor]]:
    processed: list[tuple[ProcessedInputs, torch.Tensor]] = []

    for trace in traces:
        turns = list(_iter_agent_turns(trace))
        if not turns:
            continue

        input_ids: list[int] = []
        assistant_mask: list[bool] = []
        old_logprobs: list[float] = []

        # Start with first prompt
        input_ids.extend(turns[0].prompt_token_ids)
        assistant_mask.extend([False] * len(turns[0].prompt_token_ids))
        old_logprobs.extend(turns[0].prompt_logprobs)

        for idx in range(len(turns) - 1):
            curr_turn = turns[idx]
            next_turn = turns[idx + 1]
            completion_ids = curr_turn.completion_token_ids
            completion_logps = curr_turn.completion_logprobs

            curr_len = len(curr_turn.prompt_token_ids)
            next_len = len(next_turn.prompt_token_ids)
            expected_end = curr_len + len(completion_ids)

            # Verify curr_prompt is prefix of next_prompt
            if curr_len > next_len or curr_turn.prompt_token_ids != next_turn.prompt_token_ids[:curr_len]:
                raise ValueError(f"Turn {idx}: prompt not a prefix of next")

            # Verify completion matches next_prompt
            if expected_end > next_len:
                raise ValueError(f"Turn {idx}: completion extends beyond next prompt")

            if completion_ids != next_turn.prompt_token_ids[curr_len:expected_end]:
                raise ValueError(f"Turn {idx}: completion doesn't match next prompt")

            # Verify logprobs match (same tokens, same context → same logprobs)
            next_prompt_logps = next_turn.prompt_logprobs[curr_len:expected_end]
            if completion_logps != next_prompt_logps:
                raise ValueError(f"Turn {idx}: completion logprobs don't match next prompt logprobs")

            # Add completion (masked) + user turn (unmasked)
            input_ids.extend(completion_ids)
            assistant_mask.extend([True] * len(completion_ids))
            old_logprobs.extend(completion_logps)

            input_ids.extend(next_turn.prompt_token_ids[expected_end:])
            assistant_mask.extend([False] * (next_len - expected_end))
            old_logprobs.extend(next_turn.prompt_logprobs[expected_end:])

        # Add final completion
        input_ids.extend(turns[-1].completion_token_ids)
        assistant_mask.extend([True] * len(turns[-1].completion_token_ids))
        old_logprobs.extend(turns[-1].completion_logprobs)

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids_tensor)
        assistant_tensor = torch.tensor(assistant_mask, dtype=torch.bool)
        old_logprob_tensor = torch.tensor(old_logprobs, dtype=torch.float32)

        images = _extract_images(trace.messages)
        pixel_values, image_grid_thw = _process_images(processor, images)

        processed_inputs: ProcessedInputs = {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask,
            "assistant_mask": assistant_tensor,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }

        processed.append((processed_inputs, old_logprob_tensor))

    return processed


def _iter_agent_turns(trace: Trace) -> Iterator[AgentTurnTokens]:
    for step in getattr(trace, "trace", []) or []:
        step_data = step.model_dump() if hasattr(step, "model_dump") else step
        if not isinstance(step_data, dict):
            continue
        if step_data.get("function.result_type") != "AgentResponse":
            continue

        raw_result = step_data.get("function.result")
        if not raw_result:
            continue
        try:
            payload = json.loads(raw_result)
        except json.JSONDecodeError:
            continue

        raw_response = payload.get("raw")
        if not isinstance(raw_response, dict):
            continue

        prompt_ids, prompt_logps = _parse_prompt_tokens(raw_response.get("prompt_logprobs"))
        completion_ids, completion_logps = _parse_completion_tokens(raw_response.get("choices"))

        if not prompt_ids or not completion_ids:
            continue

        yield AgentTurnTokens(prompt_ids, prompt_logps, completion_ids, completion_logps)


def _parse_prompt_tokens(data: Any) -> tuple[list[int], list[float]]:
    ids: list[int] = []
    logprobs: list[float] = []
    if not isinstance(data, list):
        return ids, logprobs
    for entry in data:
        if not entry:
            continue
        token_id_str, info = next(iter(entry.items()))
        try:
            token_id = int(token_id_str)
        except ValueError:
            continue
        ids.append(token_id)
        logprob = 0.0
        if isinstance(info, dict) and info.get("logprob") is not None:
            logprob = float(info["logprob"])
        logprobs.append(logprob)
    return ids, logprobs


def _parse_completion_tokens(choices: Any) -> tuple[list[int], list[float]]:
    ids: list[int] = []
    logprobs: list[float] = []
    if not isinstance(choices, list) or not choices:
        return ids, logprobs
    logprob_block = choices[0].get("logprobs")
    if not isinstance(logprob_block, dict):
        return ids, logprobs
    content = logprob_block.get("content")
    if not isinstance(content, list):
        return ids, logprobs
    for item in content:
        if not isinstance(item, dict):
            continue
        label = item.get("token")
        if not isinstance(label, str) or not label.startswith("token_id:"):
            continue
        try:
            token_id = int(label.split(":", 1)[1])
        except ValueError:
            continue
        ids.append(token_id)
        logprob = item.get("logprob")
        logprobs.append(float(logprob) if logprob is not None else 0.0)
    return ids, logprobs


def _extract_images(messages: list[Any]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "user":
            continue
        for block in message.get("content", []) or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "image_url":
                continue
            image_url = block.get("image_url", {})
            url = image_url.get("url", "")
            if isinstance(url, str) and url.startswith("data:image"):
                data = url.split(",", 1)[1] if "," in url else url
                images.append(_b64_to_pil(data))
    return images


def _process_images(processor: Union[ProcessorMixin, Any], images: list[Image.Image]) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not images:
        return None, None
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None:
        image_inputs = image_processor(images=images, return_tensors="pt")  # type: ignore[call-arg]
        return image_inputs.get("pixel_values"), image_inputs.get("image_grid_thw")
    return None, None


def _b64_to_pil(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")
