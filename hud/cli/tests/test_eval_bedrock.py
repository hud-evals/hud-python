"""Tests for AWS Bedrock auto-detection in hud.cli.eval."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import typer

from hud.cli.eval import EvalConfig
from hud.types import AgentType


class TestBedrockAutoDetection:
    VALID_ARN = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/my-profile"

    def test_get_agent_kwargs_detects_bedrock_arn_from_config_checkpoint_name(self) -> None:
        """Regression: ARN in [claude].checkpoint_name should trigger Bedrock client."""
        cfg = EvalConfig(
            agent_type=AgentType.CLAUDE,
            model=None,  # no CLI --model
            agent_config={"claude": {"checkpoint_name": self.VALID_ARN}},
        )

        with (
            patch("hud.settings.settings.aws_access_key_id", "AKIATEST"),
            patch("hud.settings.settings.aws_secret_access_key", "secret"),
            patch("hud.settings.settings.aws_region", "us-east-1"),
            patch("anthropic.AsyncAnthropicBedrock", return_value=MagicMock()) as mock_bedrock,
        ):
            kwargs = cfg.get_agent_kwargs()

        assert kwargs.get("checkpoint_name") == self.VALID_ARN
        assert "model_client" in kwargs
        mock_bedrock.assert_called_once()

    def test_get_agent_kwargs_bedrock_arn_missing_aws_creds_exits(self) -> None:
        """Should fail fast if ARN is detected but AWS creds are missing."""
        cfg = EvalConfig(
            agent_type=AgentType.CLAUDE,
            model=None,
            agent_config={"claude": {"checkpoint_name": self.VALID_ARN}},
        )

        with (
            patch("hud.settings.settings.aws_access_key_id", None),
            patch("hud.settings.settings.aws_secret_access_key", None),
            patch("hud.settings.settings.aws_region", None),
            pytest.raises(typer.Exit),
        ):
            cfg.get_agent_kwargs()
