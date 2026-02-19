"""Shared fixtures for the llm4eln-digest test suite."""

from __future__ import annotations

import textwrap

import pytest

from llm4eln_digest.utils.config import AppConfig, ModelConfig, ProviderConfig


@pytest.fixture()
def mock_model_config() -> ModelConfig:
    """Return a dummy ModelConfig."""
    return ModelConfig(name="Test Model", value="test-model-v1")


@pytest.fixture()
def mock_provider_config(mock_model_config: ModelConfig) -> ProviderConfig:
    """Return a ProviderConfig with dummy env vars and one model."""
    return ProviderConfig(
        key="test_provider",
        display_name="Test Provider",
        client_type="anthropic",
        env_vars={"api_key": "sk-test-key", "endpoint": "https://test.example.com"},
        models=(mock_model_config,),
    )


@pytest.fixture()
def mock_app_config(mock_provider_config: ProviderConfig) -> AppConfig:
    """Return an AppConfig containing a single provider."""
    return AppConfig(providers={"test_provider": mock_provider_config})


@pytest.fixture()
def tmp_config_yml(tmp_path: object) -> object:
    """Write a minimal valid config.yml to a temp directory and return its path."""
    from pathlib import Path

    p = Path(str(tmp_path)) / "config.yml"
    p.write_text(
        textwrap.dedent("""\
            providers:
              dummy:
                display_name: "Dummy"
                client_type: "anthropic"
                models:
                  - name: "Model A"
                    value: "model-a"
        """),
    )
    return p


@pytest.fixture()
def tmp_config_yml_with_env(tmp_path: object) -> object:
    """Write a config.yml that uses ``${ENV_VAR}`` placeholders."""
    from pathlib import Path

    p = Path(str(tmp_path)) / "config.yml"
    p.write_text(
        textwrap.dedent("""\
            providers:
              dummy:
                display_name: "Dummy"
                client_type: "anthropic"
                env_vars:
                  api_key: "${TEST_API_KEY}"
                  endpoint: "${TEST_ENDPOINT}"
                models:
                  - name: "Model A"
                    value: "model-a"
        """),
    )
    return p
