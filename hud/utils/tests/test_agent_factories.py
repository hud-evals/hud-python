from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_create_openai_agent():
    from hud.agents.openai_chat_generic import GenericOpenAIChatAgent
    from hud.utils.agent_factories import create_openai_agent

    agent = create_openai_agent(
        api_key="test_key", model_name="test_model", completion_kwargs={"temperature": 0.5}
    )
    assert isinstance(agent, GenericOpenAIChatAgent)
    assert agent.model_name == "GenericOpenAI"
    assert agent.checkpoint_name == "test_model"
    assert agent.completion_kwargs["temperature"] == 0.5


def test_create_grounded_agent():
    with (
        patch("hud.utils.agent_factories.AsyncOpenAI") as mock_async_openai,
        patch("hud.utils.agent_factories.GrounderConfig"),
        patch("hud.utils.agent_factories.GroundedOpenAIChatAgent") as mock_agent_class,
    ):
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent

        from hud.utils.agent_factories import create_grounded_agent

        agent = create_grounded_agent(
            api_key="test_key",
            grounder_api_key="grounder_key",
            model_name="test_model",
        )

        assert agent == mock_agent
        mock_async_openai.assert_called_with(api_key="test_key", base_url=None)
        mock_agent_class.assert_called_once()


def test_create_grounded_agent_custom_grounder():
    with (
        patch("hud.utils.agent_factories.AsyncOpenAI"),
        patch("hud.utils.agent_factories.GrounderConfig") as mock_grounder_config,
        patch("hud.utils.agent_factories.GroundedOpenAIChatAgent"),
    ):
        from hud.utils.agent_factories import create_grounded_agent

        create_grounded_agent(
            api_key="test_key",
            grounder_api_key="grounder_key",
            model_name="test_model",
            grounder_api_base="https://custom.api",
            grounder_model="custom/model",
        )

        mock_grounder_config.assert_called_with(
            api_base="https://custom.api",
            model="custom/model",
            api_key="grounder_key",
        )
