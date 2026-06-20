"""Direct Fireworks Training API RL loop over HUD-style arithmetic tasks.

This is intentionally close to ``cookbooks/rl-training``'s preview task:
sample answers for multiplication prompts, grade locally, then train with a
GRPO-style objective using Fireworks' managed trainer/deployment service.

The loop does not use Fireworks native datasets or RFT jobs. It uses the direct
Training API:

1. ``FiretitanServiceClient.from_firetitan_config(...)``
2. ``DeploymentSampler`` for high-parallel rollouts
3. ``forward_backward_custom(...)`` + ``optim_step(...)``
4. ``save_weights_for_sampler(...)`` + sampler refresh
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tinker
import torch
from dotenv import load_dotenv
from fireworks.training.sdk import (
    AdaptiveConcurrencyController,
    FiretitanServiceClient,
    GradAccNormalization,
)
from openai import AsyncOpenAI
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_MODEL = "accounts/fireworks/models/qwen3-8b"
DEFAULT_TOKENIZER_MODEL = "Qwen/Qwen3-8B"
DEFAULT_TRAINING_SHAPE = "accounts/fireworks/trainingShapes/qwen3-8b-128k"
DEFAULT_INFERENCE_BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_INFERENCE_MODEL = "accounts/fireworks/models/gpt-oss-120b"


@dataclass(frozen=True, slots=True)
class ArithmeticTask:
    group_index: int
    a: int
    b: int

    @property
    def expected(self) -> int:
        return self.a * self.b

    @property
    def prompt(self) -> str:
        return f"What is {self.a} * {self.b}? Reply with only the integer."


@dataclass(slots=True)
class RolloutRecord:
    task: ArithmeticTask
    text: str
    reward: float
    tokens: list[int]
    rollout_logprobs: list[float]
    loss_weights: torch.Tensor


def load_env() -> None:
    """Load the repo-level .env so FIREWORKS_API_KEY is available in cookbooks."""
    load_dotenv(ROOT / ".env")
    load_dotenv()


def make_tasks(
    *, groups: int, seed: int, min_a: int, max_a: int, min_b: int, max_b: int
) -> list[ArithmeticTask]:
    rng = random.Random(seed)
    return [
        ArithmeticTask(
            group_index=i,
            a=rng.randint(min_a, max_a),
            b=rng.randint(min_b, max_b),
        )
        for i in range(groups)
    ]


def format_prompt_tokens(tokenizer: Any, prompt: str, *, enable_thinking: bool = False) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    # Hybrid reasoning models (e.g. Qwen3) default to a <think> block. For a
    # direct-answer task with a tight token budget that reasoning never reaches
    # the integer, so rewards collapse to zero. enable_thinking flows into the
    # chat template (ignored by templates that don't define it).
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return list(tokenizer.encode(text))


def grade_answer(text: str, expected: int) -> tuple[float, int | None]:
    integers = re.findall(r"-?\d+", text)
    got = int(integers[-1]) if integers else None
    return (1.0 if got == expected else 0.0), got


async def sample_one(
    sampler: Any,
    tokenizer: Any,
    task: ArithmeticTask,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool,
) -> RolloutRecord:
    prompt_tokens = format_prompt_tokens(tokenizer, task.prompt, enable_thinking=enable_thinking)
    completions = await sampler.sample_with_prompt_tokens(
        prompt_tokens,
        n=1,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        logprobs=True,
    )
    completion = completions[0]
    tokens = list(completion.full_tokens)
    prompt_len = int(completion.prompt_len)
    output_len = max(0, len(tokens) - prompt_len)
    output_logprobs = list(completion.inference_logprobs)
    text = str(completion.text)
    reward, _got = grade_answer(text, task.expected)
    model_input_len = max(0, len(tokens) - 1)
    rollout_logprobs = [0.0] * max(0, prompt_len - 1) + output_logprobs[:output_len]
    if len(rollout_logprobs) < model_input_len:
        rollout_logprobs.extend([0.0] * (model_input_len - len(rollout_logprobs)))
    else:
        rollout_logprobs = rollout_logprobs[:model_input_len]
    weights = torch.zeros(model_input_len, dtype=torch.float32)
    if output_len:
        weights[max(0, prompt_len - 1) :] = 1.0
    return RolloutRecord(
        task=task,
        text=text,
        reward=reward,
        tokens=tokens,
        rollout_logprobs=rollout_logprobs,
        loss_weights=weights,
    )


async def sample_rollouts(
    sampler: Any,
    tokenizer: Any,
    tasks: list[ArithmeticTask],
    *,
    rollouts_per_prompt: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool,
) -> list[RolloutRecord]:
    jobs = [
        sample_one(
            sampler,
            tokenizer,
            task,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=enable_thinking,
        )
        for task in tasks
        for _ in range(rollouts_per_prompt)
    ]
    return await asyncio.gather(*jobs)


async def sample_one_inference(
    client: AsyncOpenAI,
    task: ArithmeticTask,
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> RolloutRecord:
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": task.prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    text = response.choices[0].message.content or ""
    reward, _got = grade_answer(text, task.expected)
    return RolloutRecord(
        task=task,
        text=text,
        reward=reward,
        tokens=[],
        rollout_logprobs=[],
        loss_weights=torch.zeros(0, dtype=torch.float32),
    )


async def sample_rollouts_inference(
    client: AsyncOpenAI,
    tasks: list[ArithmeticTask],
    *,
    model: str,
    rollouts_per_prompt: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    parallelism: int,
) -> list[RolloutRecord]:
    sem = asyncio.Semaphore(parallelism)

    async def run_one(task: ArithmeticTask) -> RolloutRecord:
        async with sem:
            return await sample_one_inference(
                client,
                task,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

    jobs = [run_one(task) for task in tasks for _ in range(rollouts_per_prompt)]
    return await asyncio.gather(*jobs)


def reward_stats(records: list[RolloutRecord]) -> dict[str, float]:
    if not records:
        return {"reward_mean": 0.0, "reward_std": 0.0, "reward_min": 0.0, "reward_max": 0.0}
    rewards = [r.reward for r in records]
    mean = sum(rewards) / len(rewards)
    variance = sum((r - mean) ** 2 for r in rewards) / max(1, len(rewards) - 1)
    return {
        "reward_mean": mean,
        "reward_std": math.sqrt(variance),
        "reward_min": min(rewards),
        "reward_max": max(rewards),
    }


def advantages_by_record(records: list[RolloutRecord]) -> list[float]:
    grouped: dict[int, list[float]] = {}
    for record in records:
        grouped.setdefault(record.task.group_index, []).append(record.reward)

    stats: dict[int, tuple[float, float]] = {}
    for group, rewards in grouped.items():
        mean = sum(rewards) / len(rewards)
        variance = sum((r - mean) ** 2 for r in rewards) / max(1, len(rewards) - 1)
        std = math.sqrt(variance)
        stats[group] = (mean, std if std > 1e-6 else 1.0)

    return [
        (record.reward - stats[record.task.group_index][0]) / stats[record.task.group_index][1]
        for record in records
    ]


def make_datums(records: list[RolloutRecord]) -> list[tinker.Datum]:
    return [
        tinker.Datum(
            model_input=tinker.ModelInput.from_ints(record.tokens[:-1]),
            loss_fn_inputs={
                "target_tokens": tinker.TensorData(
                    data=record.tokens[1:],
                    dtype="int64",
                    shape=[len(record.tokens) - 1],
                ),
                "weights": tinker.TensorData(
                    data=record.loss_weights.tolist(),
                    dtype="float32",
                    shape=[len(record.tokens) - 1],
                ),
            },
        )
        for record in records
    ]


def make_grpo_loss(records: list[RolloutRecord], advantages: list[float]):
    rollout_logprobs = [
        torch.tensor(record.rollout_logprobs, dtype=torch.float32) for record in records
    ]
    advantage_tensors = [torch.tensor(value, dtype=torch.float32) for value in advantages]

    def loss_fn(
        data: list[tinker.Datum], logprobs_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total_loss = torch.tensor(0.0)
        total_tokens = 0.0
        ratios: list[float] = []

        for i, logprobs in enumerate(logprobs_list):
            weights = torch.tensor(data[i].loss_fn_inputs["weights"].data, dtype=torch.float32)
            min_len = min(len(logprobs), len(weights), len(rollout_logprobs[i]))
            if min_len == 0:
                continue
            pi = logprobs[:min_len].float()
            old = rollout_logprobs[i][:min_len]
            mask = weights[:min_len]
            ratio = torch.exp((pi - old).clamp(-8.0, 8.0))
            clipped = torch.clamp(ratio, 0.8, 1.2)
            surrogate = torch.minimum(
                ratio * advantage_tensors[i],
                clipped * advantage_tensors[i],
            )
            total_loss = total_loss - torch.dot(surrogate, mask)
            total_tokens += float(mask.sum().item())
            if mask.sum().item() > 0:
                ratios.append(float((ratio * mask).sum().item() / mask.sum().item()))

        mean_ratio = sum(ratios) / len(ratios) if ratios else 0.0
        return total_loss, {
            "policy_loss_sum": float(total_loss.item()),
            "tokens": total_tokens,
            "mean_ratio": mean_ratio,
        }

    return loss_fn


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, sort_keys=True) + "\n")


def maybe_plot(metrics_path: Path, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    rows = [
        json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line
    ]
    if not rows:
        return
    plottable = [row for row in rows if row.get("phase") in {"calibrate", "train"}]
    steps = [row["step"] for row in plottable]
    rewards = [row["reward_mean"] for row in plottable]
    losses = [row.get("policy_loss_sum", 0.0) for row in plottable]
    if not steps:
        return
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(steps, rewards, marker="o", label="reward_mean", color="tab:green")
    ax1.set_xlabel("step")
    ax1.set_ylabel("reward_mean", color="tab:green")
    ax1.set_ylim(-0.05, 1.05)
    ax2 = ax1.twinx()
    ax2.plot(steps, losses, marker="x", label="policy_loss_sum", color="tab:blue")
    ax2.set_ylabel("policy_loss_sum", color="tab:blue")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)


async def run(args: argparse.Namespace) -> None:
    load_env()
    api_key = os.environ["FIREWORKS_API_KEY"]
    output_dir = Path(args.output_dir)
    metrics_path = output_dir / "metrics.jsonl"
    plot_path = output_dir / "reward_loss.png"
    if metrics_path.exists() and not args.resume_metrics:
        metrics_path.unlink()

    if args.calibrate_only and args.calibration_backend == "inference":
        client = AsyncOpenAI(api_key=api_key, base_url=args.inference_base_url)
        tasks = make_tasks(
            groups=args.groups_per_step,
            seed=args.seed,
            min_a=args.min_a,
            max_a=args.max_a,
            min_b=args.min_b,
            max_b=args.max_b,
        )
        t0 = time.perf_counter()
        records = await sample_rollouts_inference(
            client,
            tasks,
            model=args.inference_model,
            rollouts_per_prompt=args.rollouts_per_prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            parallelism=args.parallelism,
        )
        row = {
            "phase": "calibrate",
            "backend": "inference",
            "step": 0,
            "num_rollouts": len(records),
            "rollout_seconds": time.perf_counter() - t0,
            **reward_stats(records),
        }
        append_jsonl(metrics_path, row)
        maybe_plot(metrics_path, plot_path)
        print(json.dumps(row, sort_keys=True), flush=True)
        return

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model, trust_remote_code=True)
    controller = AdaptiveConcurrencyController(initial_window=args.parallelism)
    service = FiretitanServiceClient.from_firetitan_config(
        api_key=api_key,
        base_url=args.base_url,
        base_model=args.base_model,
        tokenizer_model=args.tokenizer_model,
        lora_rank=args.lora_rank,
        training_shape_id=args.training_shape,
        deployment_id=args.deployment_id,
        learning_rate=args.learning_rate,
        replica_count=args.replicas,
        cleanup_trainer_on_close=not args.keep_trainer,
        cleanup_deployment_on_close=None if args.keep_deployment else "scale_to_zero",
    )

    try:
        training_client = None
        if not args.calibrate_only:
            training_client = service.create_training_client(
                base_model=args.base_model,
                lora_rank=args.lora_rank,
            )

        sampler = service.create_deployment_sampler(
            tokenizer=tokenizer,
            concurrency_controller=controller,
        )
        tasks = make_tasks(
            groups=args.groups_per_step,
            seed=args.seed,
            min_a=args.min_a,
            max_a=args.max_a,
            min_b=args.min_b,
            max_b=args.max_b,
        )

        for step in range(args.steps if not args.calibrate_only else 1):
            t0 = time.perf_counter()
            records = await sample_rollouts(
                sampler,
                tokenizer,
                tasks,
                rollouts_per_prompt=args.rollouts_per_prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                enable_thinking=args.enable_thinking,
            )
            rollout_seconds = time.perf_counter() - t0
            stats = reward_stats(records)
            for record in records[: args.debug_samples]:
                prompt_len = len(
                    format_prompt_tokens(
                        tokenizer, record.task.prompt, enable_thinking=args.enable_thinking
                    )
                )
                print(
                    "[sample] reward=%s output_tokens=%d text=%r"
                    % (record.reward, max(0, len(record.tokens) - prompt_len), record.text),
                    flush=True,
                )
            row: dict[str, Any] = {
                "phase": "calibrate" if args.calibrate_only else "train",
                "step": step,
                "num_rollouts": len(records),
                "rollout_seconds": rollout_seconds,
                "trainer_job_id": getattr(service, "trainer_job_id", None),
                "deployment_id": getattr(service, "deployment_id", None),
                **stats,
            }

            if args.calibrate_only:
                append_jsonl(metrics_path, row)
                maybe_plot(metrics_path, plot_path)
                print(json.dumps(row, sort_keys=True), flush=True)
                continue

            assert training_client is not None
            datums = make_datums(records)
            advantages = advantages_by_record(records)
            loss_fn = make_grpo_loss(records, advantages)
            fb_future = await training_client.forward_backward_custom_async(datums, loss_fn)
            fb = await fb_future.result_async()
            optim_future = await training_client.optim_step_async(
                tinker.AdamParams(
                    learning_rate=args.learning_rate,
                    beta1=0.9,
                    beta2=0.999,
                    eps=1e-8,
                    weight_decay=args.weight_decay,
                ),
                grad_accumulation_normalization=GradAccNormalization.NUM_LOSS_TOKENS,
            )
            await optim_future.result_async()
            row.update(fb.metrics)

            saved_future = await training_client.save_weights_for_sampler_async(f"step-{step:05d}")
            saved = await saved_future.result_async()
            row["checkpoint"] = saved.path
            sampler = service.create_deployment_sampler(
                model_path=saved.path,
                tokenizer=tokenizer,
                concurrency_controller=controller,
            )
            append_jsonl(metrics_path, row)
            maybe_plot(metrics_path, plot_path)
            print(json.dumps(row, sort_keys=True), flush=True)
    finally:
        service.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url", default=os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--inference-model", default=DEFAULT_INFERENCE_MODEL)
    parser.add_argument("--tokenizer-model", default=DEFAULT_TOKENIZER_MODEL)
    parser.add_argument("--training-shape", default=DEFAULT_TRAINING_SHAPE)
    parser.add_argument("--deployment-id", default="hud-fireworks-rl-preview")
    parser.add_argument("--output-dir", default="runs/fireworks-rl-preview")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--groups-per-step", type=int, default=8)
    parser.add_argument("--rollouts-per-prompt", type=int, default=8)
    parser.add_argument("--parallelism", type=int, default=32)
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-a", type=int, default=10)
    parser.add_argument("--max-a", type=int, default=99)
    parser.add_argument("--min-b", type=int, default=2)
    parser.add_argument("--max-b", type=int, default=9)
    parser.add_argument("--debug-samples", type=int, default=0)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Keep the model's <think> reasoning block. Off by default so direct-answer "
        "rollouts reach the integer within the token budget (hybrid models like Qwen3 "
        "otherwise spend the whole budget reasoning and score zero).",
    )
    parser.add_argument("--calibrate-only", action="store_true")
    parser.add_argument(
        "--calibration-backend",
        choices=("inference", "managed"),
        default="inference",
        help="Use Fireworks OpenAI-compatible inference for cheap calibration, or the managed Training API deployment sampler.",
    )
    parser.add_argument("--inference-base-url", default=DEFAULT_INFERENCE_BASE_URL)
    parser.add_argument("--keep-trainer", action="store_true")
    parser.add_argument("--keep-deployment", action="store_true")
    parser.add_argument("--resume-metrics", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
