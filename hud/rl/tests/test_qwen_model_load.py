"""Test coverage for ``hud.rl.model.get_model`` without distributed setup."""

from typing import Tuple

import pytest
import torch

from hud.rl.config import ModelConfig
from hud.rl.model import get_model


Bundle = Tuple[ModelConfig, torch.nn.Module]


@pytest.fixture(scope="module")
def lm_bundle() -> Bundle:
    config = ModelConfig(base_model="Qwen/Qwen2.5-0.5B")
    try:
        model = get_model(config)
    except OSError as exc:  # Offline failure
        pytest.skip(f"{config.base_model} not available locally: {exc}")
    return config, model


@pytest.fixture(scope="module")
def vl_bundle() -> Bundle:
    config = ModelConfig(base_model="Qwen/Qwen2.5-VL-3B-Instruct")
    try:
        model = get_model(config)
    except OSError as exc:
        pytest.skip(f"{config.base_model} not available locally: {exc}")
    return config, model


def test_qwen_model_can_be_loaded(lm_bundle: Bundle, vl_bundle: Bundle) -> None:
    """Ensure ``get_model`` can construct both LM and VL variants."""
    for config, model in (lm_bundle, vl_bundle):
        params = list(model.parameters())
        assert params, "Model should expose parameters"
        expected_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        assert params[0].dtype == expected_dtype, (
            f"Model {config.base_model} parameters should default to {expected_dtype}"
        )


def test_get_model_respects_freeze_vision_tower(monkeypatch) -> None:
    """``freeze_vision_tower`` should be invoked only when requested."""
    import hud.rl.model as model_module

    calls: list[str] = []
    original = model_module.freeze_vision_tower

    def tracking_freeze(model):
        calls.append("called")
        return original(model)

    monkeypatch.setattr(model_module, "freeze_vision_tower", tracking_freeze)
    try:
        get_model(ModelConfig(base_model="Qwen/Qwen2.5-VL-3B-Instruct", freeze_vision_tower=True))
    except OSError as exc:
        pytest.skip(f"Qwen VL model unavailable locally: {exc}")
    assert calls, "freeze_vision_tower should run when freeze_vision_tower=True"

    calls.clear()

    def sentinel(_model):
        calls.append("unexpected")

    monkeypatch.setattr(model_module, "freeze_vision_tower", sentinel)
    try:
        get_model(ModelConfig(base_model="Qwen/Qwen2.5-VL-3B-Instruct", freeze_vision_tower=False))
    except OSError as exc:
        pytest.skip(f"Qwen VL model unavailable locally: {exc}")
    assert not calls, "freeze_vision_tower should not run when freeze_vision_tower=False"
