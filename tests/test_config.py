"""Tests for llm4eln_digest.utils.config."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from llm4eln_digest.utils.config import (
    AppConfig,
    ModelConfig,
    ProviderConfig,
    _find_config_path,
    _resolve_env_var,
    load_config,
)

# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------


class TestModelConfig:
    def test_str_returns_name(self, mock_model_config: ModelConfig) -> None:
        assert str(mock_model_config) == "Test Model"

    def test_frozen(self, mock_model_config: ModelConfig) -> None:
        with pytest.raises(AttributeError):
            mock_model_config.name = "changed"  # type: ignore[misc]

    def test_hashable(self, mock_model_config: ModelConfig) -> None:
        d = {mock_model_config: 1}
        assert d[mock_model_config] == 1

    def test_equality(self) -> None:
        a = ModelConfig(name="m", value="v")
        b = ModelConfig(name="m", value="v")
        assert a == b


# ---------------------------------------------------------------------------
# ProviderConfig
# ---------------------------------------------------------------------------


class TestProviderConfig:
    def test_value_property(self, mock_provider_config: ProviderConfig) -> None:
        assert mock_provider_config.value == "test_provider"

    def test_eq_by_key(self) -> None:
        a = ProviderConfig(key="k", display_name="A", client_type="x", env_vars={}, models=())
        b = ProviderConfig(key="k", display_name="B", client_type="y", env_vars={}, models=())
        assert a == b

    def test_hash_by_key(self) -> None:
        a = ProviderConfig(key="k", display_name="A", client_type="x", env_vars={}, models=())
        b = ProviderConfig(key="k", display_name="B", client_type="y", env_vars={}, models=())
        assert hash(a) == hash(b)

    def test_str_returns_display_name(self, mock_provider_config: ProviderConfig) -> None:
        assert str(mock_provider_config) == "Test Provider"

    def test_eq_different_display_name_same_key(self) -> None:
        a = ProviderConfig(key="same", display_name="Name1", client_type="x", env_vars={}, models=())
        b = ProviderConfig(key="same", display_name="Name2", client_type="x", env_vars={}, models=())
        assert a == b


# ---------------------------------------------------------------------------
# AppConfig
# ---------------------------------------------------------------------------


class TestAppConfig:
    def test_default_provider(self, mock_app_config: AppConfig) -> None:
        provider = mock_app_config.default_provider
        assert provider.key == "test_provider"


# ---------------------------------------------------------------------------
# _resolve_env_var
# ---------------------------------------------------------------------------


class TestResolveEnvVar:
    def test_resolves_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_VAR", "hello")
        assert _resolve_env_var("${MY_VAR}") == "hello"

    def test_raises_for_missing_env_var(self) -> None:
        with pytest.raises(ValueError, match="not set"):
            _resolve_env_var("${DEFINITELY_MISSING_12345}")

    def test_plain_string_unchanged(self) -> None:
        assert _resolve_env_var("plain") == "plain"

    def test_multiple_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("A", "1")
        monkeypatch.setenv("B", "2")
        assert _resolve_env_var("${A}-${B}") == "1-2"


# ---------------------------------------------------------------------------
# _find_config_path
# ---------------------------------------------------------------------------


class TestFindConfigPath:
    def test_env_var_override(self, tmp_config_yml: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM4ELN_CONFIG_PATH", str(tmp_config_yml))
        assert _find_config_path() == tmp_config_yml

    def test_env_var_invalid_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM4ELN_CONFIG_PATH", "/nonexistent/path/config.yml")
        with pytest.raises(FileNotFoundError, match="non-existent file"):
            _find_config_path()

    def test_no_config_found(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv("LLM4ELN_CONFIG_PATH", raising=False)
        # Patch __file__ to point into a temp tree with no config.yml
        fake_file = tmp_path / "pkg" / "config.py"
        fake_file.parent.mkdir(parents=True)
        fake_file.touch()
        with (
            patch("llm4eln_digest.utils.config.__file__", str(fake_file)),
            pytest.raises(FileNotFoundError, match="not found"),
        ):
            _find_config_path()


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_loads_valid_config(self, tmp_config_yml: Path) -> None:
        cfg = load_config(tmp_config_yml)
        assert "dummy" in cfg.providers
        assert cfg.providers["dummy"].display_name == "Dummy"
        assert cfg.providers["dummy"].models[0].value == "model-a"

    def test_missing_providers_key(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yml"
        p.write_text("foo: bar\n")
        with pytest.raises(ValueError, match="providers"):
            load_config(p)

    def test_non_mapping_provider(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yml"
        p.write_text(
            textwrap.dedent("""\
                providers:
                  bad_prov: "just a string"
            """),
        )
        with pytest.raises(TypeError, match="must be a mapping"):
            load_config(p)

    def test_provider_with_no_models(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yml"
        p.write_text(
            textwrap.dedent("""\
                providers:
                  empty:
                    display_name: "Empty"
                    client_type: "x"
                    models: []
            """),
        )
        with pytest.raises(ValueError, match="at least one model"):
            load_config(p)

    def test_empty_providers(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yml"
        p.write_text("providers: {}\n")
        with pytest.raises(ValueError, match="at least one provider"):
            load_config(p)

    def test_model_value_used_as_name_when_omitted(self, tmp_path: Path) -> None:
        p = tmp_path / "cfg.yml"
        p.write_text(
            textwrap.dedent("""\
                providers:
                  prov:
                    display_name: "P"
                    client_type: "x"
                    models:
                      - value: "some-model"
            """),
        )
        cfg = load_config(p)
        model = cfg.providers["prov"].models[0]
        assert model.name == "some-model"
        assert model.value == "some-model"

    def test_resolves_env_vars(self, tmp_config_yml_with_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_API_KEY", "resolved-key")
        monkeypatch.setenv("TEST_ENDPOINT", "https://resolved.example.com")
        cfg = load_config(tmp_config_yml_with_env)
        env = cfg.providers["dummy"].env_vars
        assert env["api_key"] == "resolved-key"
        assert env["endpoint"] == "https://resolved.example.com"
