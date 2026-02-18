"""AI interface for interacting with various LLM providers."""

import os
from collections.abc import AsyncGenerator
from enum import Enum
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_openai import AzureChatOpenAI


class AnthropicModel(Enum):
    """Anthropic model options."""

    CLAUDE_SONNET_4_5 = "claude-sonnet-4-5"
    CLAUDE_OPUS_4_5 = "claude-opus-4-5"
    CLAUDE_HAIKU_4_5 = "claude-haiku-4-5"


class AzureOpenAiModel(Enum):
    """Azure OpenAI model options."""

    GPT_4O = "gpt-4o-2024-11-20"
    GPT_5_1 = "gpt-5.1-2025-11-13"


class ModelProvider(Enum):
    """Supported model providers."""

    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"


class AiInterface:
    """Interface for interacting with AI models.

    This class provides a unified interface for working with different LLM providers,
    supporting streaming responses, tool usage, and conversation history management.

    Attributes:
        model: The underlying chat model instance
        provider: The model provider being used
        tools: List of tools available to the model
        conversation_history: List of messages in the conversation
        system_message: Optional system message for the conversation
    """

    def __init__(
        self,
        provider: ModelProvider,
        model_name: str | AnthropicModel | AzureOpenAiModel,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[BaseTool] | None = None,
        system_message: str | None = None,
    ) -> None:
        """Initialize the AI interface.

        Args:
            provider: The model provider to use
            model_name: The specific model to use (string or enum)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response
            tools: Optional list of LangChain tools
            system_message: Optional system message for the conversation
        """
        self.provider = provider
        self.tools = tools or []
        self.conversation_history: list[BaseMessage] = []
        self.system_message = system_message

        # Extract model name from enum if needed
        model_name_str = model_name.value if isinstance(model_name, (AnthropicModel, AzureOpenAiModel)) else model_name

        # Initialize the appropriate model
        base_model = self._initialize_model(
            provider=provider,
            model_name=model_name_str,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Bind tools to model if provided
        self.model: Runnable[Any, Any] = base_model.bind_tools(self.tools) if self.tools else base_model

    def _initialize_model(
        self,
        provider: ModelProvider,
        model_name: str,
        temperature: float,
        max_tokens: int,
    ) -> BaseChatModel:
        """Initialize the chat model based on provider.

        Args:
            provider: The model provider
            model_name: The model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            Initialized chat model

        Raises:
            ValueError: If provider is not supported or required env vars are missing
        """
        if provider == ModelProvider.ANTHROPIC:
            api_key = os.getenv("ANTHROPIC_FOUNDRY_API_KEY")
            if not api_key:
                msg = "ANTHROPIC_FOUNDRY_API_KEY environment variable is required"
                raise ValueError(msg)

            endpoint = os.getenv("ENDPOINT")
            if not endpoint:
                msg = "ENDPOINT environment variable is required for Anthropic"
                raise ValueError(msg)

            return ChatAnthropic(  # type: ignore[call-arg]
                model_name=model_name,
                temperature=temperature,
                max_tokens_to_sample=max_tokens,
                anthropic_api_key=api_key,
                base_url=endpoint,
            )

        elif provider == ModelProvider.AZURE_OPENAI:
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION")

            if not api_key or not endpoint:
                msg = "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables are required"
                raise ValueError(msg)

            return AzureChatOpenAI(  # type: ignore[call-arg]
                azure_deployment=model_name,
                temperature=temperature,
                openai_api_key=api_key,
                azure_endpoint=endpoint,
                openai_api_version=api_version,
            )

        else:
            msg = f"Unsupported provider: {provider}"
            raise ValueError(msg)

    def add_tool(self, tool: BaseTool) -> None:
        """Add a tool to the interface.

        Args:
            tool: LangChain tool to add
        """
        self.tools.append(tool)
        # Rebind all tools to the model
        if hasattr(self.model, "bind_tools"):
            self.model = self.model.bind_tools(self.tools)

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []

    def _build_messages(self, user_message: str) -> list[BaseMessage]:
        """Build the message list for the model.

        Args:
            user_message: The user's message

        Returns:
            List of messages including system, history, and current user message
        """
        messages: list[BaseMessage] = []

        # Add system message if provided
        if self.system_message:
            messages.append(SystemMessage(content=self.system_message))

        # Add conversation history
        messages.extend(self.conversation_history)

        # Add current user message
        messages.append(HumanMessage(content=user_message))

        return messages

    async def get_response(self, user_message: str, stream: bool = True) -> str | AsyncGenerator[str, None]:
        """Get a response from the AI model.

        Args:
            user_message: The user's input message
            stream: Whether to stream the response

        Returns:
            Either a complete response string or an async generator for streaming

        Raises:
            Exception: If the model invocation fails
        """
        # Build messages
        messages = self._build_messages(user_message)

        # Add user message to history
        self.conversation_history.append(HumanMessage(content=user_message))

        if stream:
            return self._stream_response(messages)
        else:
            # Non-streaming response
            response = await self.model.ainvoke(messages)
            response_text = response.content if isinstance(response.content, str) else str(response.content)

            # Add AI response to history
            self.conversation_history.append(AIMessage(content=response_text))

            return response_text

    async def _stream_response(self, messages: list[BaseMessage]) -> AsyncGenerator[str, None]:
        """Stream response from the model.

        Args:
            messages: List of messages to send to the model

        Yields:
            Chunks of the response text
        """
        full_response = ""

        async for chunk in self.model.astream(messages):
            if hasattr(chunk, "content") and chunk.content:
                content = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                full_response += content
                yield content

        # Add complete response to history
        self.conversation_history.append(AIMessage(content=full_response))

    async def get_response_with_tools(self, user_message: str) -> dict[str, Any]:
        """Get a response that may include tool calls.

        Args:
            user_message: The user's input message

        Returns:
            Dictionary containing response text and any tool calls

        Raises:
            Exception: If the model invocation fails
        """
        messages = self._build_messages(user_message)

        # Add user message to history
        self.conversation_history.append(HumanMessage(content=user_message))

        # Invoke model
        response = await self.model.ainvoke(messages)

        # Extract response content and tool calls
        response_text = response.content if isinstance(response.content, str) else str(response.content)
        tool_calls = getattr(response, "tool_calls", [])

        # Add AI response to history
        self.conversation_history.append(AIMessage(content=response_text, tool_calls=tool_calls))

        return {
            "text": response_text,
            "tool_calls": tool_calls,
        }


def create_anthropic_interface(
    model: AnthropicModel = AnthropicModel.CLAUDE_SONNET_4_5,
    temperature: float = 0.7,
    system_message: str | None = None,
    tools: list[BaseTool] | None = None,
) -> AiInterface:
    """Create an AI interface for Anthropic models.

    Args:
        model: The Anthropic model to use
        temperature: Sampling temperature
        system_message: Optional system message
        tools: Optional list of tools

    Returns:
        Configured AiInterface instance
    """
    return AiInterface(
        provider=ModelProvider.ANTHROPIC,
        model_name=model,
        temperature=temperature,
        tools=tools,
        system_message=system_message,
    )


def create_azure_interface(
    model: AzureOpenAiModel = AzureOpenAiModel.GPT_4O,
    temperature: float = 0.7,
    system_message: str | None = None,
    tools: list[BaseTool] | None = None,
) -> AiInterface:
    """Create an AI interface for Azure OpenAI models.

    Args:
        model: The Azure OpenAI model to use
        temperature: Sampling temperature
        system_message: Optional system message
        tools: Optional list of tools

    Returns:
        Configured AiInterface instance
    """
    return AiInterface(
        provider=ModelProvider.AZURE_OPENAI,
        model_name=model,
        temperature=temperature,
        tools=tools,
        system_message=system_message,
    )
