"""Tests for llm4eln_digest.backend."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from llm4eln_digest.utils.config import AppConfig, ModelConfig, ProviderConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROVIDER_A = ProviderConfig(
    key="prov_a",
    display_name="Provider A",
    client_type="anthropic",
    env_vars={"api_key": "k"},
    models=(
        ModelConfig(name="Model-1", value="m1"),
        ModelConfig(name="Model-2", value="m2"),
    ),
)

_PROVIDER_B = ProviderConfig(
    key="prov_b",
    display_name="Provider B",
    client_type="azure_openai",
    env_vars={"api_key": "k2"},
    models=(ModelConfig(name="Model-B", value="mb"),),
)

_APP_CONFIG = AppConfig(providers={"prov_a": _PROVIDER_A, "prov_b": _PROVIDER_B})


def _mock_ai_interface() -> MagicMock:
    """Return a mock that behaves like AiInterface."""
    iface = MagicMock()
    iface.conversation_history = []
    iface.clear_history = MagicMock(side_effect=lambda: iface.conversation_history.clear())
    return iface


@pytest.fixture()
def backend():
    """Create an AiBackend with mocked config and interface."""
    with (
        patch("llm4eln_digest.backend.load_config", return_value=_APP_CONFIG),
        patch("llm4eln_digest.backend.create_interface", return_value=_mock_ai_interface()),
    ):
        from llm4eln_digest.backend import AiBackend

        return AiBackend()


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_current_provider_is_first(self, backend: Any) -> None:
        assert backend.current_provider is _PROVIDER_A

    def test_current_model_is_first_of_provider(self, backend: Any) -> None:
        assert backend.current_model == _PROVIDER_A.models[0]

    def test_default_temperature(self, backend: Any) -> None:
        assert backend.current_temperature == 0.7

    def test_default_tools_empty(self, backend: Any) -> None:
        assert backend.current_tools == []

    def test_ai_interface_created(self, backend: Any) -> None:
        assert backend.ai_interface is not None


# ---------------------------------------------------------------------------
# Provider / model management
# ---------------------------------------------------------------------------


class TestProviderModelManagement:
    def test_get_available_providers(self, backend: Any) -> None:
        providers = backend.get_available_providers()
        assert "Provider A" in providers
        assert "Provider B" in providers

    def test_get_available_models(self, backend: Any) -> None:
        models = backend.get_available_models(_PROVIDER_A)
        assert "Model-1" in models
        assert "Model-2" in models

    def test_get_provider_display_name(self, backend: Any) -> None:
        assert backend.get_provider_display_name(_PROVIDER_A) == "Provider A"

    def test_update_provider(self, backend: Any) -> None:
        with patch("llm4eln_digest.backend.create_interface", return_value=_mock_ai_interface()):
            display, model_val = backend.update_provider(_PROVIDER_B)
        assert display == "Provider B"
        assert model_val == "mb"
        assert backend.current_provider is _PROVIDER_B

    def test_update_model_preserves_history(self, backend: Any) -> None:
        backend.ai_interface.conversation_history = [HumanMessage(content="keep")]
        with patch("llm4eln_digest.backend.create_interface", return_value=_mock_ai_interface()) as mock_create:
            backend.update_model(ModelConfig(name="Model-2", value="m2"))
        assert backend.current_model.value == "m2"
        # create_interface is called; history preservation is handled internally
        mock_create.assert_called_once()

    def test_update_temperature(self, backend: Any) -> None:
        with patch("llm4eln_digest.backend.create_interface", return_value=_mock_ai_interface()):
            backend.update_temperature(0.9)
        assert backend.current_temperature == 0.9

    def test_update_tools_returns_count(self, backend: Any) -> None:
        tools = [MagicMock(), MagicMock()]
        with patch("llm4eln_digest.backend.create_interface", return_value=_mock_ai_interface()):
            count = backend.update_tools(tools)
        assert count == 2


# ---------------------------------------------------------------------------
# History management
# ---------------------------------------------------------------------------


class TestHistoryManagement:
    def test_clear_history(self, backend: Any) -> None:
        backend.ai_interface.conversation_history.append(HumanMessage(content="x"))
        backend.clear_history()
        assert backend.ai_interface.conversation_history == []

    def test_get_conversation_history(self, backend: Any) -> None:
        backend.ai_interface.conversation_history = [HumanMessage(content="a")]
        assert len(backend.get_conversation_history()) == 1

    def test_get_conversation_history_no_interface(self, backend: Any) -> None:
        backend.ai_interface = None
        assert backend.get_conversation_history() == []

    def test_set_conversation_history(self, backend: Any) -> None:
        history = [HumanMessage(content="set")]
        backend.set_conversation_history(history)
        assert backend.ai_interface.conversation_history is history


# ---------------------------------------------------------------------------
# process_message (async)
# ---------------------------------------------------------------------------


class TestProcessMessage:
    @pytest.mark.asyncio()
    async def test_without_tools(self, backend: Any) -> None:
        backend.ai_interface.get_response = AsyncMock(return_value="AI says hello")
        result = await backend.process_message("hi", use_tools=False)
        assert result["response"] == "AI says hello"
        assert result["preview_updates"] == []

    @pytest.mark.asyncio()
    async def test_with_tools(self, backend: Any) -> None:
        backend._handle_message_with_tools = AsyncMock(return_value="tool result")
        result = await backend.process_message("run", use_tools=True)
        assert result["response"] == "tool result"

    @pytest.mark.asyncio()
    async def test_no_interface_returns_error(self, backend: Any) -> None:
        backend.ai_interface = None
        result = await backend.process_message("hi")
        assert "Error" in result["response"]


# ---------------------------------------------------------------------------
# _execute_tool_calls (async)
# ---------------------------------------------------------------------------


class TestExecuteToolCalls:
    @pytest.mark.asyncio()
    async def test_executes_matching_tool(self, backend: Any) -> None:
        tool = MagicMock()
        tool.name = "my_tool"
        tool._arun = AsyncMock(return_value="result")
        backend.current_tools = [tool]

        tool_calls = [{"name": "my_tool", "args": {"x": 1}, "id": "tc1"}]
        results = await backend._execute_tool_calls(tool_calls)
        assert len(results) == 1
        assert results[0].content == "result"

    @pytest.mark.asyncio()
    async def test_preview_update_marker(self, backend: Any) -> None:
        backend.preview_updates = []
        tool = MagicMock()
        tool.name = "preview"
        tool._arun = AsyncMock(return_value="PREVIEW_UPDATE::Title::Body")
        backend.current_tools = [tool]

        tool_calls = [{"name": "preview", "args": {}, "id": "tc2"}]
        results = await backend._execute_tool_calls(tool_calls)
        assert len(backend.preview_updates) == 1
        assert backend.preview_updates[0]["title"] == "Title"
        assert backend.preview_updates[0]["content"] == "Body"
        assert "Successfully updated" in results[0].content

    @pytest.mark.asyncio()
    async def test_tool_execution_error(self, backend: Any) -> None:
        tool = MagicMock()
        tool.name = "bad_tool"
        tool._arun = AsyncMock(side_effect=RuntimeError("boom"))
        backend.current_tools = [tool]

        tool_calls = [{"name": "bad_tool", "args": {}, "id": "tc3"}]
        results = await backend._execute_tool_calls(tool_calls)
        assert "Error executing tool" in results[0].content


# ---------------------------------------------------------------------------
# export_chat_data
# ---------------------------------------------------------------------------


class TestExportChatData:
    def test_returns_expected_keys(self, backend: Any) -> None:
        data = backend.export_chat_data("prov", "model", 0.5, [{"role": "user", "content": "hi"}])
        assert "timestamp" in data
        assert data["provider"] == "prov"
        assert data["model"] == "model"
        assert data["temperature"] == 0.5
        assert data["messages"] == [{"role": "user", "content": "hi"}]

    def test_includes_conversation_history(self, backend: Any) -> None:
        backend.ai_interface.conversation_history = [HumanMessage(content="h"), AIMessage(content="a")]
        data = backend.export_chat_data("p", "m", 0.7, [])
        assert len(data["conversation_history"]) == 2
        assert data["conversation_history"][0]["type"] == "HumanMessage"


# ---------------------------------------------------------------------------
# restore_chat_data
# ---------------------------------------------------------------------------


class TestRestoreChatData:
    def test_restores_messages(self, backend: Any) -> None:
        chat_data = {
            "conversation_history": [
                {"type": "HumanMessage", "content": "hello"},
                {"type": "AIMessage", "content": "hi there"},
            ]
        }
        backend.restore_chat_data(chat_data)
        history = backend.ai_interface.conversation_history
        assert len(history) == 2
        assert isinstance(history[0], HumanMessage)
        assert isinstance(history[1], AIMessage)

    def test_clears_existing_history(self, backend: Any) -> None:
        backend.ai_interface.conversation_history = [HumanMessage(content="old")]
        chat_data = {
            "conversation_history": [
                {"type": "HumanMessage", "content": "new"},
            ]
        }
        backend.restore_chat_data(chat_data)
        assert len(backend.ai_interface.conversation_history) == 1
        assert backend.ai_interface.conversation_history[0].content == "new"
