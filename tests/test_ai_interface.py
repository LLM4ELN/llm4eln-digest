"""Tests for llm4eln_digest.utils.ai_interface."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from llm4eln_digest.utils.ai_interface import AiInterface, create_interface
from llm4eln_digest.utils.config import ModelConfig, ProviderConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(client_type: str = "anthropic") -> ProviderConfig:
    return ProviderConfig(
        key="test",
        display_name="Test",
        client_type=client_type,
        env_vars={"api_key": "k", "endpoint": "https://e"},
        models=(ModelConfig(name="m", value="m-v1"),),
    )


def _mock_base_model() -> MagicMock:
    """Return a MagicMock that behaves like a BaseChatModel."""
    m = MagicMock()
    m.bind_tools = MagicMock(return_value=m)
    return m


def _patch_registry(client_type: str = "anthropic") -> tuple:
    """Patch PROVIDER_CLASS_REGISTRY so the given *client_type* maps to a mock factory.

    Returns (patcher, mock_factory, mock_model) — enter the patcher as a
    context manager.
    """
    mock_model = _mock_base_model()
    mock_factory = MagicMock(return_value=mock_model)

    patcher = patch.dict(
        "llm4eln_digest.utils.ai_interface.PROVIDER_CLASS_REGISTRY",
        {client_type: mock_factory},
    )
    return patcher, mock_factory, mock_model


# ---------------------------------------------------------------------------
# _initialize_model
# ---------------------------------------------------------------------------


class TestInitializeModel:
    def test_dispatches_anthropic(self) -> None:
        provider = _make_provider("anthropic")
        patcher, mock_factory, _ = _patch_registry("anthropic")
        with patcher:
            AiInterface._initialize_model(provider, "m", 0.5, 1024)
        mock_factory.assert_called_once_with(provider, "m", 0.5, 1024)

    def test_dispatches_azure_openai(self) -> None:
        provider = _make_provider("azure_openai")
        patcher, mock_factory, _ = _patch_registry("azure_openai")
        with patcher:
            AiInterface._initialize_model(provider, "m", 0.5, 1024)
        mock_factory.assert_called_once_with(provider, "m", 0.5, 1024)

    def test_raises_for_unsupported_type(self) -> None:
        provider = _make_provider("unknown_llm")
        with pytest.raises(ValueError, match="Unsupported client_type"):
            AiInterface._initialize_model(provider, "m", 0.5, 1024)


# ---------------------------------------------------------------------------
# AiInterface.__init__
# ---------------------------------------------------------------------------


class TestAiInterfaceInit:
    def test_accepts_model_config(self) -> None:
        provider = _make_provider()
        mc = ModelConfig(name="n", value="v")
        patcher, _, _ = _patch_registry("anthropic")
        with patcher:
            iface = AiInterface(provider=provider, model_name=mc)
        assert iface.provider is provider

    def test_accepts_string_model_name(self) -> None:
        provider = _make_provider()
        patcher, _, _ = _patch_registry("anthropic")
        with patcher:
            iface = AiInterface(provider=provider, model_name="some-model")
        assert iface.provider is provider

    def test_binds_tools(self) -> None:
        provider = _make_provider()
        fake_tool = MagicMock()
        patcher, _, mock_model = _patch_registry("anthropic")
        with patcher:
            AiInterface(provider=provider, model_name="m", tools=[fake_tool])
        mock_model.bind_tools.assert_called_once_with([fake_tool])

    def test_no_bind_when_tools_empty(self) -> None:
        provider = _make_provider()
        patcher, _, mock_model = _patch_registry("anthropic")
        with patcher:
            AiInterface(provider=provider, model_name="m", tools=[])
        mock_model.bind_tools.assert_not_called()


# ---------------------------------------------------------------------------
# _build_messages
# ---------------------------------------------------------------------------


class TestBuildMessages:
    def _make_iface(self, **kwargs: object) -> AiInterface:
        provider = _make_provider()
        patcher, _, _ = _patch_registry("anthropic")
        with patcher:
            return AiInterface(provider=provider, model_name="m", **kwargs)  # type: ignore[arg-type]

    def test_includes_system_message(self) -> None:
        iface = self._make_iface(system_message="Be helpful")
        msgs = iface._build_messages("hi")
        assert isinstance(msgs[0], SystemMessage)
        assert msgs[0].content == "Be helpful"

    def test_excludes_system_when_none(self) -> None:
        iface = self._make_iface(system_message=None)
        msgs = iface._build_messages("hi")
        assert not any(isinstance(m, SystemMessage) for m in msgs)

    def test_appends_history(self) -> None:
        iface = self._make_iface()
        iface.conversation_history = [HumanMessage(content="prev"), AIMessage(content="resp")]
        msgs = iface._build_messages("new")
        assert msgs[0].content == "prev"
        assert msgs[1].content == "resp"
        assert msgs[-1].content == "new"

    def test_ends_with_human_message(self) -> None:
        iface = self._make_iface()
        msgs = iface._build_messages("hello")
        assert isinstance(msgs[-1], HumanMessage)


# ---------------------------------------------------------------------------
# clear_history
# ---------------------------------------------------------------------------


class TestClearHistory:
    def test_empties_history(self) -> None:
        provider = _make_provider()
        patcher, _, _ = _patch_registry("anthropic")
        with patcher:
            iface = AiInterface(provider=provider, model_name="m")
        iface.conversation_history = [HumanMessage(content="x")]
        iface.clear_history()
        assert iface.conversation_history == []


# ---------------------------------------------------------------------------
# add_tool
# ---------------------------------------------------------------------------


class TestAddTool:
    def test_appends_tool(self) -> None:
        provider = _make_provider()
        patcher, _, mock_model = _patch_registry("anthropic")
        with patcher:
            iface = AiInterface(provider=provider, model_name="m")
        # model is the mock, which has bind_tools
        iface.model = mock_model
        fake_tool = MagicMock()
        iface.add_tool(fake_tool)
        assert fake_tool in iface.tools

    def test_rebinds_tools(self) -> None:
        provider = _make_provider()
        patcher, _, mock_model = _patch_registry("anthropic")
        with patcher:
            iface = AiInterface(provider=provider, model_name="m")
        iface.model = mock_model
        fake_tool = MagicMock()
        iface.add_tool(fake_tool)
        mock_model.bind_tools.assert_called_with([fake_tool])


# ---------------------------------------------------------------------------
# get_response (async, mocked)
# ---------------------------------------------------------------------------


class TestGetResponse:
    @pytest.mark.asyncio()
    async def test_non_streaming_returns_string(self) -> None:
        provider = _make_provider()
        patcher, _, _ = _patch_registry("anthropic")
        with patcher:
            iface = AiInterface(provider=provider, model_name="m")

        mock_response = MagicMock()
        mock_response.content = "Hello from AI"
        iface.model = MagicMock()
        iface.model.ainvoke = AsyncMock(return_value=mock_response)

        result = await iface.get_response("test", stream=False)
        assert result == "Hello from AI"
        assert len(iface.conversation_history) == 2

    @pytest.mark.asyncio()
    async def test_streaming_returns_async_generator(self) -> None:
        provider = _make_provider()
        patcher, _, _ = _patch_registry("anthropic")
        with patcher:
            iface = AiInterface(provider=provider, model_name="m")

        result = await iface.get_response("test", stream=True)
        assert hasattr(result, "__aiter__")


# ---------------------------------------------------------------------------
# get_response_with_tools (async, mocked)
# ---------------------------------------------------------------------------


class TestGetResponseWithTools:
    @pytest.mark.asyncio()
    async def test_returns_dict_with_expected_keys(self) -> None:
        provider = _make_provider()
        patcher, _, _ = _patch_registry("anthropic")
        with patcher:
            iface = AiInterface(provider=provider, model_name="m")

        mock_response = MagicMock()
        mock_response.content = "tool response"
        mock_response.tool_calls = [{"name": "t", "args": {}, "id": "call_1"}]
        iface.model = MagicMock()
        iface.model.ainvoke = AsyncMock(return_value=mock_response)

        result = await iface.get_response_with_tools("run tool")
        assert "text" in result
        assert "tool_calls" in result
        assert result["text"] == "tool response"

    @pytest.mark.asyncio()
    async def test_adds_to_history(self) -> None:
        provider = _make_provider()
        patcher, _, _ = _patch_registry("anthropic")
        with patcher:
            iface = AiInterface(provider=provider, model_name="m")

        mock_response = MagicMock()
        mock_response.content = "ok"
        mock_response.tool_calls = []
        iface.model = MagicMock()
        iface.model.ainvoke = AsyncMock(return_value=mock_response)

        await iface.get_response_with_tools("hi")
        assert len(iface.conversation_history) == 2


# ---------------------------------------------------------------------------
# create_interface factory
# ---------------------------------------------------------------------------


class TestCreateInterface:
    def test_returns_ai_interface(self) -> None:
        provider = _make_provider()
        patcher, _, _ = _patch_registry("anthropic")
        with patcher:
            iface = create_interface(provider=provider, model="m")
        assert isinstance(iface, AiInterface)

    def test_passes_parameters(self) -> None:
        provider = _make_provider()
        patcher, _, _ = _patch_registry("anthropic")
        with patcher:
            iface = create_interface(
                provider=provider,
                model="m",
                temperature=0.3,
                max_tokens=512,
                system_message="sys",
            )
        assert iface.system_message == "sys"
