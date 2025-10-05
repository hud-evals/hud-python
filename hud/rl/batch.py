import copy
from typing import Sequence

import torch

from hud.rl.types import ProcessedInputs, TrainingSample

def batch_samples(
    samples: Sequence[TrainingSample],
    mini_batch_size: int,
    num_gpus: int,
    pad_token_id: int,
) -> list[list[TrainingSample]]:
    sample_list = list(samples)
    if not sample_list:
        return []

    minibatches = _prepare_minibatches(sample_list, mini_batch_size, pad_token_id)

    if num_gpus == 1:
        return [minibatches]

    num_padding = num_gpus - (len(minibatches) % num_gpus)

    if num_padding < num_gpus and len(minibatches) > 0:
        dummy_batch = _create_dummy_batch(minibatches[0])
        minibatches.extend([dummy_batch] * num_padding)

    per_gpu_count = len(minibatches) // num_gpus
    batches_per_gpu: list[list[TrainingSample]] = []

    for _ in range(num_gpus):
        gpu_batches: list[TrainingSample] = []
        for _ in range(per_gpu_count):
            gpu_batches.append(minibatches.pop(0))
        batches_per_gpu.append(gpu_batches)

    return batches_per_gpu


def _create_dummy_batch(reference_batch: TrainingSample) -> TrainingSample:
    dummy = copy.deepcopy(reference_batch)
    dummy.advantage = torch.zeros_like(dummy.advantage)
    dummy.inputs["assistant_mask"] = torch.zeros_like(
        dummy.inputs["assistant_mask"], dtype=torch.bool
    )
    return dummy


def _prepare_minibatches(
    samples: list[TrainingSample],
    mini_batch_size: int,
    pad_token_id: int,
) -> list[TrainingSample]:
    batched_samples: list[TrainingSample] = []
    for i in range(0, len(samples), mini_batch_size):
        chunk = samples[i : i + mini_batch_size]
        batched_samples.append(_merge_chunk(chunk, pad_token_id))

    return batched_samples


def _merge_chunk(chunk: list[TrainingSample], pad_token_id: int) -> TrainingSample:
    max_seq_len = max(sample.inputs["input_ids"].size(-1) for sample in chunk)

    for sample in chunk:
        sample.inputs["input_ids"] = torch.nn.functional.pad(
            sample.inputs["input_ids"],
            (0, max_seq_len - sample.inputs["input_ids"].size(-1)),
            value=pad_token_id,
        )
        sample.inputs["attention_mask"] = torch.nn.functional.pad(
            sample.inputs["attention_mask"],
            (0, max_seq_len - sample.inputs["attention_mask"].size(-1)),
            value=0,
        )
        sample.inputs["assistant_mask"] = torch.nn.functional.pad(
            sample.inputs["assistant_mask"],
            (0, max_seq_len - sample.inputs["assistant_mask"].size(-1)),
            value=False,
        )

    input_ids = torch.stack([sample.inputs["input_ids"] for sample in chunk], dim=0)
    attention_mask = torch.stack([sample.inputs["attention_mask"] for sample in chunk], dim=0)
    assistant_mask = torch.stack([sample.inputs["assistant_mask"] for sample in chunk], dim=0)

    merged_inputs: ProcessedInputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "assistant_mask": assistant_mask,
        "pixel_values": _merge_pixel_values(chunk),
        "image_grid_thw": _merge_image_grids(chunk),
    }

    advantages = torch.cat([sample.advantage.view(-1) for sample in chunk], dim=0)
    temperatures = torch.cat([sample.temperature.view(-1) for sample in chunk], dim=0)
    return TrainingSample(inputs=merged_inputs, advantage=advantages, temperature=temperatures)


def _merge_pixel_values(chunk: list[TrainingSample]) -> torch.Tensor | None:
    pixel_values_list = [sample.inputs.get("pixel_values") for sample in chunk]

    if any(pv is None for pv in pixel_values_list):
        return None

    return torch.cat([pv for pv in pixel_values_list if pv is not None], dim=0)


def _merge_image_grids(chunk: list[TrainingSample]) -> torch.Tensor | None:
    grids_list = [sample.inputs.get("image_grid_thw") for sample in chunk]

    if any(grid is None for grid in grids_list):
        return None

    return torch.cat([grid for grid in grids_list if grid is not None], dim=0)
