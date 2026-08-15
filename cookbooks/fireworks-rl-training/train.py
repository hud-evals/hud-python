"""Train a Fireworks LoRA directly from rewards produced by a HUD taskset."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import tinker
from dotenv import load_dotenv
from fireworks.training.sdk import FiretitanServiceClient
from hud.agents.base import Agent
from hud.agents.types import AgentStep, Sample, Usage
from hud.eval import HUDRuntime, LocalRuntime, Provider, Taskset
from hud.eval.run import Run
from tinker_cookbook.renderers import get_renderer, get_text_content
from tinker_cookbook.tokenizer_utils import get_tokenizer

from env import multiply

HERE = Path(__file__).resolve().parent
SERVERLESS_URL = "https://api.fireworks.ai/training/v1/serverless"
DEFAULT_BASE_MODEL = "accounts/fireworks/models/qwen3p5-9b"
DEFAULT_TOKENIZER_MODEL = "Qwen/Qwen3.5-9B"
# Thinking mode spends the whole token budget on a <think> block and never
# reaches the final answer line, so rewards collapse to zero on this task.
DEFAULT_RENDERER = "qwen3_5_disable_thinking"


def serverless_url(value: str) -> str:
    """Accept the Fireworks API root or the complete serverless endpoint."""
    root = value.rstrip("/")
    if root.endswith("/training/v1/serverless"):
        return root
    if root.endswith("/training/v1"):
        return f"{root}/serverless"
    return f"{root}/training/v1/serverless"


def group_relative_advantages(rewards: list[float]) -> list[float]:
    if len(rewards) < 2:
        return [0.0] * len(rewards)
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / (len(rewards) - 1)
    std = math.sqrt(variance)
    if std < 1e-8:
        return [0.0] * len(rewards)
    return [(reward - mean) / std for reward in rewards]


def within_group_reward_std(runs: list[Run]) -> float:
    """Mean per-group reward std: the spread GRPO actually trains on.

    Advantages are computed within each group, so a group whose rollouts all
    score the same produces zero gradient even if the overall mean looks
    healthy.
    """
    grouped: dict[str, list[float]] = {}
    for run in runs:
        if run.group_id:
            grouped.setdefault(run.group_id, []).append(run.reward)
    stds = []
    for rewards in grouped.values():
        if len(rewards) < 2:
            continue
        mean = sum(rewards) / len(rewards)
        variance = sum((reward - mean) ** 2 for reward in rewards) / (len(rewards) - 1)
        stds.append(math.sqrt(variance))
    return sum(stds) / len(stds) if stds else 0.0


def report_calibration(runs: list[Run], *, debug_samples: int) -> None:
    completed = [run for run in runs if _sample(run) is not None]
    rewards = [run.reward for run in completed]
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    print(
        f"calibration: {len(completed)}/{len(runs)} rollouts completed "
        f"reward_mean={mean_reward:.3f} "
        f"within_group_reward_std={within_group_reward_std(completed):.3f}",
        flush=True,
    )
    for run in completed[:debug_samples]:
        sample = _sample(run)
        assert sample is not None
        text = (run.trace.content or "").strip()
        if len(text) > 400:
            text = text[:400] + "..."
        print(
            f"--- reward={run.reward:.2f} output_tokens={len(sample.output_token_ids)}\n{text}",
            flush=True,
        )


def make_taskset(*, count: int, seed: int, a: tuple[int, int], b: tuple[int, int]) -> Taskset:
    # The default 3-digit x 3-digit lands mid-difficulty for the 9B model:
    # right often but not always, so groups keep the reward spread GRPO
    # trains on. Tune the ranges with --min-a/--max-a/--min-b/--max-b.
    a_count = a[1] - a[0] + 1
    b_count = b[1] - b[0] + 1
    population = a_count * b_count
    if count > population:
        raise ValueError(
            f"requested {count} arithmetic tasks, but the operand ranges contain "
            f"only {population} unique pairs"
        )

    indices = random.Random(seed).sample(range(population), count)
    return Taskset(
        f"arithmetic-{seed}",
        [
            multiply(
                a=a[0] + index // b_count,
                b=b[0] + index % b_count,
            )
            for index in indices
        ],
    )


def split_taskset(
    source: Taskset,
    *,
    train_count: int,
    eval_count: int,
    seed: int,
) -> tuple[Taskset, Taskset]:
    """Create deterministic, disjoint train and evaluation subsets."""
    tasks = list(source)
    required = train_count + eval_count
    if len(tasks) < required:
        raise ValueError(
            f"task source {source.name!r} has {len(tasks)} tasks; "
            f"need at least {required} for {train_count} training and {eval_count} evaluation tasks"
        )

    random.Random(seed).shuffle(tasks)
    train = Taskset(f"{source.name}-train", tasks[:train_count], origin=source.origin)
    evaluation = Taskset(
        f"{source.name}-eval",
        tasks[train_count:required],
        origin=source.origin,
    )
    return train, evaluation


def resolve_rollout_source(
    args: argparse.Namespace,
) -> tuple[Taskset, Taskset, Provider | HUDRuntime]:
    """Resolve bundled, local-file, or hosted HUD tasks and their runtime."""
    eval_count = 0 if getattr(args, "calibrate", False) else args.eval_tasks
    if args.taskset:
        source = Taskset.from_api(args.taskset)
        train, evaluation = split_taskset(
            source,
            train_count=args.tasks_per_step,
            eval_count=eval_count,
            seed=args.seed,
        )
        return train, evaluation, HUDRuntime()

    if args.tasks_file:
        source = Taskset.from_file(args.tasks_file)
        train, evaluation = split_taskset(
            source,
            train_count=args.tasks_per_step,
            eval_count=eval_count,
            seed=args.seed,
        )
        return train, evaluation, LocalRuntime(str(Path(args.env_path).resolve()))

    source = make_taskset(
        count=args.tasks_per_step + eval_count,
        seed=args.seed,
        a=(args.min_a, args.max_a),
        b=(args.min_b, args.max_b),
    )
    train, evaluation = split_taskset(
        source,
        train_count=args.tasks_per_step,
        eval_count=eval_count,
        seed=args.seed,
    )
    return train, evaluation, LocalRuntime(str(HERE / "env.py"))


class FireworksAgent(Agent):
    """One-turn HUD agent backed by an in-session Fireworks sampler snapshot."""

    def __init__(
        self,
        *,
        sampler: Any,
        renderer: Any,
        model: str,
        max_tokens: int,
        temperature: float,
        timeout: float,
        max_seq_len: int,
    ) -> None:
        self.sampler = sampler
        self.renderer = renderer
        self.model = model
        self.timeout = timeout
        self.max_seq_len = max_seq_len
        self.sampling_params = tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=renderer.get_stop_sequences(),
        )

    async def __call__(self, run: Run) -> None:
        prompt = self.renderer.build_generation_prompt(
            [{"role": "user", "content": run.prompt_text}]
        )
        if prompt.length + self.sampling_params.max_tokens > self.max_seq_len:
            raise ValueError(
                "rendered prompt plus max_tokens exceeds max_seq_len "
                f"({prompt.length} + {self.sampling_params.max_tokens} > {self.max_seq_len}); "
                "lower --max-tokens or raise --max-seq-len"
            )

        future = self.sampler.sample(
            prompt=prompt,
            num_samples=1,
            sampling_params=self.sampling_params,
        )
        result = await asyncio.to_thread(future.result, timeout=self.timeout)
        if not result.sequences:
            raise RuntimeError("Fireworks sampler returned no completion")

        sequence = result.sequences[0]
        output_tokens = list(sequence.tokens)
        output_logprobs = list(sequence.logprobs or [])
        if not output_tokens or len(output_tokens) != len(output_logprobs):
            raise RuntimeError("Fireworks sampler did not return aligned tokens and logprobs")

        content = get_text_content(self.renderer.parse_response(output_tokens)[0])
        prompt_tokens = list(prompt.to_ints())
        run.record(
            AgentStep(
                content=content,
                done=True,
                model=self.model,
                sample=Sample(
                    prompt_token_ids=prompt_tokens,
                    output_token_ids=output_tokens,
                    output_logprobs=[float(value) for value in output_logprobs],
                ),
                usage=Usage(
                    prompt_tokens=len(prompt_tokens),
                    completion_tokens=len(output_tokens),
                ),
            )
        )
        run.trace.content = content
        run.trace.status = "completed"
        run.trace.stop_reason = "done"


def _sample(run: Run) -> Sample | None:
    return run.trace.final(
        lambda step: step.sample if isinstance(step, AgentStep) and step.sample else None
    )


def make_training_batch(runs: list[Run]) -> tuple[list[tinker.Datum], int]:
    """Convert valid HUD rollout groups into Fireworks importance-sampling datums."""
    grouped: dict[str, list[Run]] = {}
    for run in runs:
        if run.group_id and _sample(run) is not None:
            grouped.setdefault(run.group_id, []).append(run)

    datums: list[tinker.Datum] = []
    kept_groups = 0
    for group_runs in grouped.values():
        rewards = [run.reward for run in group_runs]
        advantages = group_relative_advantages(rewards)
        if not any(advantages):
            continue
        kept_groups += 1

        for run, advantage in zip(group_runs, advantages, strict=True):
            sample = _sample(run)
            assert sample is not None
            if not sample.prompt_token_ids or not sample.output_token_ids:
                continue

            # Position i predicts token i + 1, so the model input is
            # prompt + response[:-1], the targets are the response shifted one
            # left, and prompt positions carry advantage 0 so only response
            # tokens contribute loss.
            prompt = tinker.ModelInput.from_ints(sample.prompt_token_ids)
            model_input = prompt.append(
                tinker.EncodedTextChunk(tokens=sample.output_token_ids[:-1])
            )
            response_start = prompt.length - 1
            output_length = len(sample.output_token_ids)
            target_tokens = [0] * response_start + sample.output_token_ids
            rollout_logprobs = [0.0] * response_start + sample.output_logprobs
            token_advantages = [0.0] * response_start + [advantage] * output_length
            datums.append(
                tinker.Datum(
                    model_input=model_input,
                    loss_fn_inputs={
                        "target_tokens": tinker.TensorData(
                            data=target_tokens,
                            dtype="int64",
                            shape=[len(target_tokens)],
                        ),
                        "logprobs": tinker.TensorData(
                            data=rollout_logprobs,
                            dtype="float32",
                            shape=[len(rollout_logprobs)],
                        ),
                        "advantages": tinker.TensorData(
                            data=token_advantages,
                            dtype="float32",
                            shape=[len(token_advantages)],
                        ),
                    },
                )
            )

    return datums, kept_groups


def mean_loss(result: Any) -> float | None:
    """Loss per input token, from the metrics the serverless trainer reports."""
    metrics = result.metrics or {}
    loss_sum = metrics.get("loss:sum")
    tokens = metrics.get("total_tokens:sum")
    if loss_sum is None or not tokens:
        return None
    return float(loss_sum) / float(tokens)


def write_metric(path: Path, metric: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(metric, sort_keys=True) + "\n")


async def run_rollouts(
    *,
    service: FiretitanServiceClient,
    training_client: Any,
    tokenizer: Any,
    renderer: Any,
    snapshot_name: str,
    taskset: Taskset,
    runtime: Provider | HUDRuntime,
    group_size: int,
    max_concurrent: int,
    max_tokens: int,
    temperature: float,
    timeout: float,
    max_seq_len: int,
) -> tuple[list[Run], str]:
    snapshot = training_client.save_weights_for_sampler(snapshot_name).result()
    if not snapshot.path:
        raise RuntimeError(f"sampler checkpoint {snapshot_name!r} returned no path")

    sampler = service.create_sampling_client(model_path=snapshot.path, tokenizer=tokenizer)
    try:
        agent = FireworksAgent(
            sampler=sampler,
            renderer=renderer,
            model=snapshot.path,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            max_seq_len=max_seq_len,
        )
        job = await taskset.run(
            agent,
            runtime=runtime,
            group=group_size,
            max_concurrent=max_concurrent,
        )
        return job.runs, snapshot.path
    finally:
        sampler.close()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "--steps": args.steps,
        "--tasks-per-step": args.tasks_per_step,
        "--max-tokens": args.max_tokens,
        "--max-concurrent": args.max_concurrent,
        "--eval-tasks": args.eval_tasks,
        "--checkpoint-every": args.checkpoint_every,
    }
    too_small = [flag for flag, value in positive.items() if value < 1]
    if too_small:
        raise SystemExit(f"{', '.join(too_small)} must be at least 1")
    if args.group_size < 2:
        raise SystemExit("--group-size must be at least 2 (GRPO needs reward spread per group)")
    if args.min_a > args.max_a or args.min_b > args.max_b:
        raise SystemExit("--min-a/--min-b must not exceed --max-a/--max-b")
    if args.debug_samples < 0:
        raise SystemExit("--debug-samples must be at least 0")
    if args.tasks_file and not args.env_path:
        raise SystemExit("--tasks-file requires --env-path")
    if args.env_path and not args.tasks_file:
        raise SystemExit("--env-path requires --tasks-file")


async def train(args: argparse.Namespace) -> None:
    load_dotenv()
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise SystemExit("Set FIREWORKS_API_KEY before running this cookbook.")

    validate_args(args)
    try:
        taskset, eval_taskset, runtime = resolve_rollout_source(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    if not args.resume_from and not args.calibrate:
        metrics_path.write_text("", encoding="utf-8")

    tokenizer = get_tokenizer(args.tokenizer_model)
    renderer = get_renderer(args.renderer, tokenizer)
    service = FiretitanServiceClient(
        api_key=api_key,
        base_url=serverless_url(args.base_url),
    )
    training_client = (
        service.create_training_client_from_state_with_optimizer(args.resume_from)
        if args.resume_from
        else service.create_lora_training_client(
            base_model=args.base_model,
            rank=args.lora_rank,
        )
    )

    session = getattr(service, "training_session_id", None)
    print(
        f"Connected to Fireworks serverless training: session={session} "
        f"run={getattr(training_client, 'run_id', None)}\n"
        f"steps={args.steps} tasks={args.tasks_per_step} "
        f"group={args.group_size} model={args.base_model}",
        flush=True,
    )

    try:
        if args.calibrate:
            runs, _ = await run_rollouts(
                service=service,
                training_client=training_client,
                tokenizer=tokenizer,
                renderer=renderer,
                snapshot_name="calibrate",
                taskset=taskset,
                runtime=runtime,
                group_size=args.group_size,
                max_concurrent=args.max_concurrent,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.sampling_timeout,
                max_seq_len=args.max_seq_len,
            )
            report_calibration(runs, debug_samples=args.debug_samples)
            return

        for step in range(1, args.steps + 1):
            started = time.perf_counter()
            runs, snapshot = await run_rollouts(
                service=service,
                training_client=training_client,
                tokenizer=tokenizer,
                renderer=renderer,
                snapshot_name=f"policy-{step:04d}",
                taskset=taskset,
                runtime=runtime,
                group_size=args.group_size,
                max_concurrent=args.max_concurrent,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.sampling_timeout,
                max_seq_len=args.max_seq_len,
            )
            datums, kept_groups = make_training_batch(runs)
            loss = None
            updated = False
            if datums:
                backward = training_client.forward_backward(datums, args.loss_fn).result()
                loss = mean_loss(backward)
                training_client.optim_step(
                    tinker.AdamParams(
                        learning_rate=args.learning_rate,
                        beta1=0.9,
                        beta2=0.95,
                        eps=1e-8,
                        weight_decay=0.0,
                    )
                ).result()
                updated = True
            elif args.require_update:
                raise RuntimeError(
                    "no rollout group had reward variation, so no optimizer update was applied; "
                    "run --calibrate, adjust task difficulty, and retry"
                )

            if step % args.checkpoint_every == 0:
                training_client.save_state(f"state-{step:04d}").result()

            completed = [run for run in runs if _sample(run) is not None]
            mean_reward = (
                sum(run.reward for run in completed) / len(completed) if completed else 0.0
            )
            metric = {
                "step": step,
                "reward": mean_reward,
                "reward_std_within_group": within_group_reward_std(completed),
                "rollouts": len(runs),
                "valid_rollouts": len(completed),
                "kept_groups": kept_groups,
                "training_datums": len(datums),
                "updated": updated,
                "loss": loss,
                "snapshot": snapshot,
                "seconds": time.perf_counter() - started,
            }
            write_metric(metrics_path, metric)
            print(
                f"step {step:02d}/{args.steps} reward={mean_reward:.3f} "
                f"kept_groups={kept_groups}/{args.tasks_per_step} "
                f"datums={len(datums)} updated={updated} "
                f"loss={loss if loss is not None else 'n/a'} "
                f"elapsed={metric['seconds']:.1f}s",
                flush=True,
            )

        final_state = training_client.save_state("final-state").result()
        eval_runs, final_snapshot = await run_rollouts(
            service=service,
            training_client=training_client,
            tokenizer=tokenizer,
            renderer=renderer,
            snapshot_name="final",
            taskset=eval_taskset,
            runtime=runtime,
            group_size=1,
            max_concurrent=args.max_concurrent,
            max_tokens=args.max_tokens,
            temperature=0.0,
            timeout=args.sampling_timeout,
            max_seq_len=args.max_seq_len,
        )
        eval_reward = sum(run.reward for run in eval_runs) / len(eval_runs)
        print(
            f"Evaluation reward={eval_reward:.3f} on {len(eval_runs)} held-out tasks\n"
            f"Sampler checkpoint: {final_snapshot}\n"
            f"Training checkpoint: {getattr(final_state, 'path', 'final-state')}\n"
            f"Metrics: {metrics_path}",
            flush=True,
        )
    finally:
        service.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("FIREWORKS_BASE_URL", SERVERLESS_URL),
        help="Fireworks API root or the complete serverless training endpoint",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="Fireworks base model enabled for serverless training on your account",
    )
    parser.add_argument(
        "--tokenizer-model",
        default=DEFAULT_TOKENIZER_MODEL,
        help="Hugging Face tokenizer matching --base-model",
    )
    parser.add_argument(
        "--renderer",
        default=DEFAULT_RENDERER,
        help="tinker-cookbook renderer name; change it together with --base-model",
    )
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA adapter rank")
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=8192,
        help="hard cap on rendered prompt plus sampled tokens",
    )
    parser.add_argument("--learning-rate", type=float, default=2.5e-5, help="Adam learning rate")
    parser.add_argument("--temperature", type=float, default=1.0, help="rollout temperature")
    parser.add_argument(
        "--sampling-timeout",
        type=float,
        default=600.0,
        help="seconds to wait for one sampling request",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="save a resumable training checkpoint every N steps",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="resume from a training checkpoint: <account>/<run-id>/state-NNNN",
    )
    parser.add_argument("--seed", type=int, default=0, help="task generation seed")
    parser.add_argument(
        "--output-dir",
        default="runs/fireworks-serverless",
        help="where metrics.jsonl is written",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--taskset",
        help="hosted HUD taskset name or id; uses HUDRuntime and requires HUD_API_KEY",
    )
    source.add_argument(
        "--tasks-file",
        help="local HUD task source (.py, directory, .json, or .jsonl); requires --env-path",
    )
    parser.add_argument(
        "--env-path",
        help="local HUD environment source used with --tasks-file",
    )
    parser.add_argument("--steps", type=int, default=30, help="optimizer steps")
    parser.add_argument(
        "--tasks-per-step", type=int, default=8, help="task groups sampled per step"
    )
    parser.add_argument(
        "--group-size", type=int, default=8, help="rollouts per task (the GRPO group)"
    )
    parser.add_argument("--max-tokens", type=int, default=1024, help="generated tokens per rollout")
    parser.add_argument("--max-concurrent", type=int, default=4, help="simultaneous rollouts")
    parser.add_argument(
        "--eval-tasks", type=int, default=16, help="held-out tasks for the final eval"
    )
    parser.add_argument(
        "--loss-fn",
        choices=("importance_sampling", "ppo", "cispo"),
        default="importance_sampling",
        help="server-side RL loss; all three take the same token/logprob/advantage datums",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="roll out one batch from the untrained adapter, report reward spread, and exit",
    )
    parser.add_argument(
        "--require-update",
        action="store_true",
        help="fail if a training step has no reward variation and cannot apply an optimizer update",
    )
    parser.add_argument(
        "--debug-samples",
        type=int,
        default=0,
        help="with --calibrate, print the first N rollouts (reward, tokens, text)",
    )
    parser.add_argument("--min-a", type=int, default=100, help="task difficulty: lower bound of a")
    parser.add_argument("--max-a", type=int, default=999, help="task difficulty: upper bound of a")
    parser.add_argument("--min-b", type=int, default=100, help="task difficulty: lower bound of b")
    parser.add_argument("--max-b", type=int, default=999, help="task difficulty: upper bound of b")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(train(parse_args()))
