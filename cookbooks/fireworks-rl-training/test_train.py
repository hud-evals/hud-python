from __future__ import annotations

import asyncio
import json
import re
from argparse import Namespace
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from hud.agents.types import AgentStep, Sample
from hud.eval import HUDRuntime, LocalRuntime, Taskset
from hud.eval.run import Run
from hud.settings import settings
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

import train as training
from env import multiply
from train import (
    group_relative_advantages,
    make_taskset,
    make_training_batch,
    resolve_rollout_source,
    serverless_url,
    split_taskset,
    within_group_reward_std,
)


@pytest.fixture(autouse=True)
def isolate_hud(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "telemetry_enabled", False)
    monkeypatch.setattr(settings, "api_key", None)
    monkeypatch.setattr(settings, "telemetry_local_dir", None)


@pytest.fixture
def tokenizer() -> PreTrainedTokenizerFast:
    vocabulary = {token: i for i, token in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    backend = Tokenizer(models.BPE(vocab=vocabulary, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token="<|im_end|>",
        additional_special_tokens=["<|im_start|>", "<think>", "</think>"],
    )


def resolved(value: object) -> Future:
    future = Future()
    future.set_result(value)
    return future


def rollout(*, group: str, reward: float) -> Run:
    run = Run(None, "multiply", {})
    run.group_id = group
    run.grade.reward = reward
    run.record(
        AgentStep(
            content="42",
            sample=Sample(
                prompt_token_ids=[1, 2],
                output_token_ids=[3, 4],
                output_logprobs=[-0.2, -0.1],
            ),
        )
    )
    return run


def test_serverless_url_accepts_root_and_complete_endpoint() -> None:
    expected = "https://api.fireworks.ai/training/v1/serverless"
    assert serverless_url("https://api.fireworks.ai") == expected
    assert serverless_url(expected) == expected


def test_within_group_reward_std_averages_per_group_spread() -> None:
    runs = [
        rollout(group="spread", reward=0.0),
        rollout(group="spread", reward=1.0),
        rollout(group="flat", reward=1.0),
        rollout(group="flat", reward=1.0),
    ]
    # sample std of [0, 1] is ~0.707; the flat group contributes 0.
    assert within_group_reward_std(runs) == pytest.approx(0.3536, abs=1e-3)
    assert within_group_reward_std([]) == 0.0


def test_group_relative_advantages_are_centered() -> None:
    advantages = group_relative_advantages([0.0, 1.0])
    assert sum(advantages) == pytest.approx(0.0)
    assert advantages[0] < 0 < advantages[1]


def test_training_batch_keeps_only_groups_with_reward_spread() -> None:
    runs = [
        rollout(group="learnable", reward=0.0),
        rollout(group="learnable", reward=1.0),
        rollout(group="flat", reward=0.0),
        rollout(group="flat", reward=0.0),
    ]

    datums, kept_groups = make_training_batch(runs)

    assert kept_groups == 1
    assert len(datums) == 2
    for datum in datums:
        assert datum.model_input.length == 3
        assert len(datum.loss_fn_inputs["target_tokens"].data) == 3
        assert len(datum.loss_fn_inputs["logprobs"].data) == 3
        assert len(datum.loss_fn_inputs["advantages"].data) == 3


def test_split_taskset_creates_disjoint_deterministic_subsets() -> None:
    source = make_taskset(count=8, seed=0, a=(10, 99), b=(10, 99))

    train, evaluation = split_taskset(source, train_count=5, eval_count=3, seed=7)
    train_again, evaluation_again = split_taskset(source, train_count=5, eval_count=3, seed=7)

    train_slugs = [task.slug for task in train]
    evaluation_slugs = [task.slug for task in evaluation]
    assert train_slugs == [task.slug for task in train_again]
    assert evaluation_slugs == [task.slug for task in evaluation_again]
    assert set(train_slugs).isdisjoint(evaluation_slugs)


def test_make_taskset_rejects_more_tasks_than_unique_operand_pairs() -> None:
    with pytest.raises(ValueError, match="only 4 unique pairs"):
        make_taskset(count=5, seed=0, a=(1, 2), b=(1, 2))


def test_resolve_rollout_source_supports_hosted_and_local_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_taskset(count=4, seed=0, a=(10, 99), b=(10, 99))
    monkeypatch.setattr(Taskset, "from_api", classmethod(lambda cls, name: source))
    monkeypatch.setattr(Taskset, "from_file", classmethod(lambda cls, path: source))
    common = {
        "tasks_per_step": 2,
        "eval_tasks": 2,
        "seed": 0,
    }

    _, _, hosted_runtime = resolve_rollout_source(
        Namespace(taskset="demo", tasks_file=None, env_path=None, **common)
    )
    _, _, local_runtime = resolve_rollout_source(
        Namespace(taskset=None, tasks_file="tasks.py", env_path="env.py", **common)
    )

    assert isinstance(hosted_runtime, HUDRuntime)
    assert isinstance(local_runtime, LocalRuntime)


def test_default_rollout_source_has_disjoint_evaluation_tasks() -> None:
    train, evaluation, runtime = resolve_rollout_source(
        Namespace(
            taskset=None,
            tasks_file=None,
            env_path=None,
            tasks_per_step=5,
            eval_tasks=3,
            seed=0,
            min_a=10,
            max_a=99,
            min_b=10,
            max_b=99,
        )
    )

    assert {task.slug for task in train}.isdisjoint(task.slug for task in evaluation)
    assert isinstance(runtime, LocalRuntime)


@pytest.fixture
def fireworks_service(tokenizer: PreTrainedTokenizerFast) -> MagicMock:
    service = MagicMock()
    client = service.create_lora_training_client.return_value
    client.save_weights_for_sampler.side_effect = lambda name: resolved(
        SimpleNamespace(path=f"test/run/{name}")
    )
    client.save_state.side_effect = lambda name: resolved(SimpleNamespace(path=f"test/run/{name}"))
    client.forward_backward.return_value = resolved(
        SimpleNamespace(metrics={"loss:sum": 1.0, "total_tokens:sum": 10})
    )
    client.optim_step.return_value = resolved(None)
    attempts: dict[str, int] = {}

    def sample(*, prompt, num_samples, sampling_params):
        text = tokenizer.decode(prompt.to_ints())
        operands = re.search(r"What is (\d+) \* (\d+)", text)
        assert operands is not None
        attempts[text] = attempts.get(text, 0) + 1
        correct = int(operands[1]) * int(operands[2])
        answer = correct if attempts[text] % 2 else correct + 1
        tokens = tokenizer.encode(f"{answer}<|im_end|>", add_special_tokens=False)
        return resolved(
            SimpleNamespace(
                sequences=[SimpleNamespace(tokens=tokens, logprobs=[-0.5] * len(tokens))]
            )
        )

    service.create_sampling_client.return_value.sample.side_effect = sample
    return service


@pytest.mark.parametrize("calibrate", [False, True])
def test_training_lifecycle_with_mocked_fireworks(
    monkeypatch, tmp_path, tokenizer, fireworks_service, calibrate
) -> None:
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setattr(training, "get_tokenizer", lambda model: tokenizer)
    factory = MagicMock(return_value=fireworks_service)
    monkeypatch.setattr(training, "FiretitanServiceClient", factory)
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--steps",
            "1",
            "--tasks-per-step",
            "1",
            "--group-size",
            "2",
            "--eval-tasks",
            "1",
            "--max-concurrent",
            "1",
            "--require-update",
            "--output-dir",
            str(tmp_path),
        ]
        + (["--calibrate"] if calibrate else []),
    )
    args = training.parse_args()
    assert args.base_model == "accounts/fireworks/models/qwen3p8-27b"
    assert args.tokenizer_model == "Qwen/Qwen3.8-27B"
    assert args.renderer == "qwen3_8_disable_thinking"

    asyncio.run(training.train(args))

    client = fireworks_service.create_lora_training_client.return_value
    fireworks_service.create_lora_training_client.assert_called_once_with(
        base_model=args.base_model, rank=8
    )
    if calibrate:
        client.forward_backward.assert_not_called()
        client.optim_step.assert_not_called()
        assert not (tmp_path / "eval-after.json").exists()
    else:
        datums, loss_fn = client.forward_backward.call_args.args
        assert loss_fn == "importance_sampling"
        assert len(datums) == 2
        assert any(value > 0 for value in datums[0].loss_fn_inputs["advantages"].data)
        assert any(value < 0 for value in datums[1].loss_fn_inputs["advantages"].data)
        client.optim_step.assert_called_once()
        client.save_state.assert_called_once_with("final-state")
        metric = json.loads((tmp_path / "metrics.jsonl").read_text())
        assert metric["updated"] is True
        assert metric["valid_rollouts"] == 2
        assert metric["reward_std_within_group"] > 0
        assert (tmp_path / "eval-after.json").exists()
    assert not (tmp_path / "eval-before.json").exists()
    assert fireworks_service.create_sampling_client.return_value.sample.call_count == (
        2 if calibrate else 3
    )
    fireworks_service.close.assert_called_once()


def test_evaluates_same_held_out_tasks_before_and_after_training(
    monkeypatch, tmp_path, tokenizer, fireworks_service
) -> None:
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setattr(training, "get_tokenizer", lambda model: tokenizer)
    monkeypatch.setattr(training, "FiretitanServiceClient", lambda **kwargs: fireworks_service)
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--steps",
            "1",
            "--tasks-per-step",
            "2",
            "--group-size",
            "2",
            "--eval-tasks",
            "3",
            "--max-concurrent",
            "1",
            "--max-tokens",
            "9",
            "--eval-before",
            "--require-update",
            "--output-dir",
            str(tmp_path),
        ],
    )
    args = training.parse_args()

    asyncio.run(training.train(args))

    requests = fireworks_service.create_sampling_client.return_value.sample.call_args_list
    assert len(requests) == 10
    before_requests, training_requests, after_requests = requests[:3], requests[3:7], requests[7:]
    before_prompts = [call.kwargs["prompt"].to_ints() for call in before_requests]
    assert before_prompts == [call.kwargs["prompt"].to_ints() for call in after_requests]
    train_prompts = [call.kwargs["prompt"].to_ints() for call in training_requests]
    assert set(map(tuple, before_prompts)).isdisjoint(map(tuple, train_prompts))
    for before, after in zip(before_requests, after_requests, strict=True):
        assert before.kwargs["sampling_params"] == after.kwargs["sampling_params"]
        assert before.kwargs["sampling_params"].temperature == 0.0
        assert before.kwargs["sampling_params"].max_tokens == args.max_tokens
    assert all(call.kwargs["sampling_params"].temperature == 1.0 for call in training_requests)

    events = [
        name.rsplit(".", 1)[-1]
        for name, _, _ in fireworks_service.mock_calls
        if name.endswith(".sample") or name.endswith(".optim_step")
    ]
    assert events == ["sample"] * 7 + ["optim_step"] + ["sample"] * 3
    client = fireworks_service.create_lora_training_client.return_value
    datums, _ = client.forward_backward.call_args.args
    assert len(datums) == 4
    for datum, request in zip(datums, training_requests, strict=True):
        prompt = request.kwargs["prompt"].to_ints()
        assert datum.model_input.to_ints()[: len(prompt)] == prompt

    before = json.loads((tmp_path / "eval-before.json").read_text())
    after = json.loads((tmp_path / "eval-after.json").read_text())
    assert before["snapshot"] == "test/run/initial"
    assert after["snapshot"] == "test/run/final"
    assert [sample["prompt"] for sample in before["samples"]] == [
        sample["prompt"] for sample in after["samples"]
    ]
    for evaluation in (before, after):
        assert evaluation["temperature"] == 0.0
        assert evaluation["max_tokens"] == args.max_tokens
        assert len(evaluation["samples"]) == 3
        for sample in evaluation["samples"]:
            operands = re.search(r"What is (\d+) \* (\d+)", sample["prompt"])
            assert operands is not None
            expected = int(operands[1]) * int(operands[2])
            assert sample["info"]["expected"] == expected
            assert sample["info"]["got"] == int(sample["answer"])
            assert sample["reward"] == float(int(sample["answer"]) == expected)
            tokens = tokenizer.encode(sample["answer"] + "<|im_end|>", add_special_tokens=False)
            assert sample["output_tokens"] == len(tokens)
            assert sample["at_token_limit"] is (len(tokens) == args.max_tokens)
    assert all(sample["reward"] == 1.0 for sample in before["samples"])
    assert all(sample["reward"] == 0.0 for sample in after["samples"])
    config_text = (tmp_path / "config.json").read_text()
    assert json.loads(config_text) == vars(args)
    assert "test-key" not in config_text
    metrics = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert len(metrics) == 1
    assert metrics[0]["training_datums"] == 4
    assert metrics[0]["rollouts"] == 4


def test_reusing_output_directory_replaces_evaluations_only_when_training(
    monkeypatch, tmp_path, tokenizer, fireworks_service
) -> None:
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setattr(training, "get_tokenizer", lambda model: tokenizer)
    monkeypatch.setattr(training, "FiretitanServiceClient", lambda **kwargs: fireworks_service)
    argv = [
        "train.py",
        "--steps",
        "1",
        "--tasks-per-step",
        "1",
        "--group-size",
        "2",
        "--eval-tasks",
        "2",
        "--max-concurrent",
        "1",
        "--output-dir",
        str(tmp_path),
    ]
    monkeypatch.setattr("sys.argv", argv + ["--eval-before"])
    asyncio.run(training.train(training.parse_args()))
    files = ["config.json", "eval-before.json", "eval-after.json", "metrics.jsonl"]
    saved = {name: (tmp_path / name).read_text() for name in files}

    monkeypatch.setattr("sys.argv", argv + ["--calibrate"])
    asyncio.run(training.train(training.parse_args()))
    assert {name: (tmp_path / name).read_text() for name in files} == saved

    monkeypatch.setattr("sys.argv", argv)
    asyncio.run(training.train(training.parse_args()))
    assert not (tmp_path / "eval-before.json").exists()
    assert (tmp_path / "eval-after.json").read_text() != saved["eval-after.json"]
    assert json.loads((tmp_path / "config.json").read_text())["eval_before"] is False
    assert fireworks_service.create_sampling_client.return_value.sample.call_count == 12


@pytest.mark.parametrize("failure", ["sampling", "grading"])
def test_rollout_failure_fails_calibration(
    monkeypatch, tmp_path, tokenizer, fireworks_service, failure
):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setattr(training, "get_tokenizer", lambda model: tokenizer)
    monkeypatch.setattr(training, "FiretitanServiceClient", lambda **kwargs: fireworks_service)
    argv = [
        "train.py",
        "--calibrate",
        "--tasks-per-step",
        "1",
        "--group-size",
        "2",
        "--output-dir",
        str(tmp_path),
    ]
    sampler = fireworks_service.create_sampling_client.return_value
    if failure == "sampling":
        future = Future()
        future.set_exception(RuntimeError("sampler unavailable"))
        sampler.sample.side_effect = None
        sampler.sample.return_value = future
    else:
        source = tmp_path / "broken.py"
        source.write_text(
            "from hud import Environment\n"
            'env = Environment("broken-grader")\n'
            "@env.template()\n"
            "async def multiply():\n"
            '    answer = yield "What is 123 * 456?"\n'
            '    raise RuntimeError("grader unavailable")\n'
            "tasks = [multiply()]\n"
        )
        argv.extend(["--tasks-file", str(source), "--env-path", str(source)])
    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(RuntimeError, match="2/2 rollouts failed"):
        asyncio.run(training.train(training.parse_args()))

    fireworks_service.create_lora_training_client.return_value.optim_step.assert_not_called()
    sampler.close.assert_called_once()
    fireworks_service.close.assert_called_once()


@pytest.mark.integration
def test_default_renderer_matches_qwen38_chat_template() -> None:
    tokenizer = training.get_tokenizer(training.DEFAULT_TOKENIZER_MODEL)
    renderer = training.get_renderer(training.DEFAULT_RENDERER, tokenizer)
    messages = [{"role": "user", "content": "What is 123 * 456?"}]
    expected = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert renderer.build_generation_prompt(messages).to_ints() == expected
    tokens = tokenizer.encode("56088<|im_end|>", add_special_tokens=False)
    message, _ = renderer.parse_response(tokens)
    assert training.get_text_content(message) == "56088"


@pytest.mark.parametrize(
    ("a", "b", "answer", "reward"),
    [
        (4861, 3217, "15637837", 1.0),
        (4861, 3217, "The final answer is:\n15,637,837", 1.0),
        (4861, 3217, "The answer is 15637837, as calculated above.", 0.0),
        (4861, 3217, "The answer is 15,637,837, as calculated above.", 0.0),
        (2, 3, "6,", 0.0),
        (4861, 3217, r"\boxed{15,637,837}", 1.0),
        (4861, 3217, r"$\boxed{15,637,837}$", 1.0),
        (4861, 3217, "**15,637,837**", 1.0),
        (4861, 3217, "`15,637,837`", 1.0),
        (4861, 3217, "Working...\n\n  15,637,837  \n \t\n", 1.0),
        (4861, 3217, "+15,637,837", 1.0),
        (4861, 3217, "15637837\n0", 0.0),
        (4861, 3217, "15637837\nThat's my answer.", 0.0),
        (4861, -3217, "-15,637,837", 1.0),
        (3, 279, "15,637,837", 0.0),
        (4861, 3217, "0.15637837", 0.0),
        (4861, 3217, "1e15637837", 0.0),
        (4861, 3217, "15,63,7837", 0.0),
        (4861, 3217, "15,63,7837,", 0.0),
        (4861, 3217, "15,,637,837,", 0.0),
        (4861, 3217, "No answer", 0.0),
        (4861, 3217, " \n\t\n", 0.0),
    ],
)
def test_grades_integer_on_last_nonempty_line(a, b, answer, reward, tokenizer, fireworks_service):
    tokens = tokenizer.encode(f"{answer}<|im_end|>", add_special_tokens=False)
    sampler = fireworks_service.create_sampling_client.return_value
    sampler.sample.side_effect = None
    sampler.sample.return_value = resolved(
        SimpleNamespace(sequences=[SimpleNamespace(tokens=tokens, logprobs=[-0.5] * len(tokens))])
    )
    agent = training.FireworksAgent(
        sampler=sampler,
        renderer=training.get_renderer(training.DEFAULT_RENDERER, tokenizer),
        model=training.DEFAULT_BASE_MODEL,
        max_tokens=2048,
        temperature=1.0,
        timeout=10,
        max_seq_len=8192,
    )
    job = asyncio.run(multiply(a=a, b=b).run(agent, runtime=LocalRuntime(training.HERE / "env.py")))
    assert job.runs[0].trace.status == "completed"
    assert job.runs[0].reward == reward
