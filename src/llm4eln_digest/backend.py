"""Backend logic for AI interface management and message processing."""

from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from llm4eln_digest.utils.ai_interface import (
    AiInterface,
    AnthropicModel,
    AzureOpenAiModel,
    ModelProvider,
    create_anthropic_interface,
    create_azure_interface,
)


class AiBackend:
    """Backend for AI interface management and message processing.

    This class handles all the business logic for:
    - AI interface creation and management
    - Provider and model configuration
    - Tool execution
    - Message processing with/without tools
    - Conversation history management
    """

    def __init__(self) -> None:
        """Initialize the AI backend."""
        # Build provider to model enum mapping
        self.provider_model_map = {
            ModelProvider.ANTHROPIC: AnthropicModel,
            ModelProvider.AZURE_OPENAI: AzureOpenAiModel,
        }

        # Build provider to interface creator mapping
        self.provider_interface_map: dict[ModelProvider, Callable[..., AiInterface]] = {
            ModelProvider.ANTHROPIC: create_anthropic_interface,
            ModelProvider.AZURE_OPENAI: create_azure_interface,
        }

        # Build provider display names
        self.provider_display_names = {
            ModelProvider.ANTHROPIC: "Anthropic Foundry",
            ModelProvider.AZURE_OPENAI: "Azure OpenAI",
        }

        # Dynamically build model options from available enums for each provider
        self.provider_models: dict[ModelProvider, dict[str, Any]] = {}
        for provider, model_enum in self.provider_model_map.items():
            self.provider_models[provider] = {self.format_model_name(model.name): model for model in model_enum}

        # Current configuration
        self.current_provider = ModelProvider.ANTHROPIC
        self.current_model = next(iter(self.provider_models[self.current_provider].values()))
        self.current_temperature = 0.7
        self.current_tools: list = []

        # AI interface
        self.ai_interface: AiInterface | None = None
        self._create_ai_interface()

    def get_available_providers(self) -> dict[str, ModelProvider]:
        """Get available providers with display names.

        Returns:
            Dictionary mapping display names to provider enums
        """
        return {self.provider_display_names[p]: p for p in ModelProvider}

    def get_available_models(self, provider: ModelProvider) -> dict[str, Any]:
        """Get available models for a specific provider.

        Args:
            provider: The provider to get models for

        Returns:
            Dictionary mapping model display names to model enums
        """
        return self.provider_models[provider]

    def get_provider_display_name(self, provider: ModelProvider) -> str:
        """Get display name for a provider.

        Args:
            provider: The provider enum

        Returns:
            Human-readable provider name
        """
        return self.provider_display_names[provider]

    @staticmethod
    def format_model_name(enum_name: str) -> str:
        """Format enum name to human-readable model name.

        Args:
            enum_name: Enum name like "SONNET_4_5" or "GPT_4O"

        Returns:
            Formatted name like "Sonnet 4.5" or "GPT-4o"
        """
        # Replace underscores with spaces and title case
        formatted = enum_name.replace("_", " ").title()

        # Special handling for version numbers: "4 5" -> "4.5", "3 5" -> "3.5", "5 1" -> "5.1"
        formatted = formatted.replace(" 4 5", " 4.5").replace(" 3 5", " 3.5").replace(" 5 1", " 5.1")

        # Special handling for GPT models
        if formatted.startswith("Gpt"):
            formatted = formatted.replace("Gpt", "GPT-", 1)
            # "GPT- 4O" -> "GPT-4o", "GPT- 35" -> "GPT-3.5"
            formatted = formatted.replace(" 4O", "4o").replace(" 35", "3.5")

        return formatted

    def _create_ai_interface(self, preserve_history: bool = True) -> None:
        """Create AI interface based on current configuration.

        Args:
            preserve_history: If True, preserve conversation history from existing interface
        """
        # Save existing conversation history if requested
        existing_history = []
        if preserve_history and self.ai_interface is not None:
            existing_history = self.ai_interface.conversation_history.copy()

        # Get the interface creator function for the current provider
        interface_creator = self.provider_interface_map[self.current_provider]

        # Create new interface using the appropriate creator function
        self.ai_interface = interface_creator(
            model=self.current_model,
            temperature=self.current_temperature,
            system_message="You are a helpful ELN assistant.",
            tools=self.current_tools,
        )

        # Restore conversation history
        if preserve_history and existing_history:
            self.ai_interface.conversation_history = existing_history

    def update_provider(self, provider: ModelProvider) -> tuple[str, str]:
        """Update the current provider and reset model.

        Args:
            provider: New provider to use

        Returns:
            Tuple of (provider_display_name, model_name) for UI notification
        """
        self.current_provider = provider

        # Update model to first available for new provider
        available_models = self.provider_models[provider]
        self.current_model = next(iter(available_models.values()))

        # Recreate interface (clear history when changing providers)
        self._create_ai_interface(preserve_history=False)

        return (
            self.provider_display_names[provider],
            self.current_model.value,
        )

    def update_model(self, model: Any) -> str:
        """Update the current model.

        Args:
            model: New model to use

        Returns:
            Model name for UI notification
        """
        self.current_model = model
        self._create_ai_interface(preserve_history=True)
        return str(model.value)

    def update_temperature(self, temperature: float) -> None:
        """Update the temperature setting.

        Args:
            temperature: New temperature value
        """
        self.current_temperature = temperature
        self._create_ai_interface(preserve_history=True)

    def update_tools(self, tools: list) -> int:
        """Update the available tools.

        Args:
            tools: List of tool instances

        Returns:
            Number of tools enabled
        """
        self.current_tools = tools
        self._create_ai_interface(preserve_history=True)
        return len(tools)

    def clear_history(self) -> None:
        """Clear the conversation history."""
        if self.ai_interface:
            self.ai_interface.clear_history()

    def get_conversation_history(self) -> list:
        """Get the current conversation history.

        Returns:
            List of conversation messages
        """
        if self.ai_interface:
            return self.ai_interface.conversation_history
        return []

    def set_conversation_history(self, history: list) -> None:
        """Set the conversation history.

        Args:
            history: List of conversation messages
        """
        if self.ai_interface:
            self.ai_interface.conversation_history = history

    async def process_message(self, message: str, use_tools: bool = False) -> dict[str, Any]:
        """Process a user message and return AI response with preview updates.

        Args:
            message: User's message
            use_tools: Whether to enable tool execution

        Returns:
            Dictionary with 'response' (AI text) and 'preview_updates' (list of preview update dicts)
        """
        if not self.ai_interface:
            return {"response": "Error: AI interface not initialized", "preview_updates": []}

        self.preview_updates: list[dict[str, str]] = []  # Reset preview updates

        if not use_tools:
            # No tools enabled, use simple response
            response = await self.ai_interface.get_response(message, stream=False)
            response_text = response if isinstance(response, str) else str(response)
            return {"response": response_text, "preview_updates": []}
        else:
            # Tools are enabled, use tool-aware response with execution loop
            response_text = await self._handle_message_with_tools(message)
            return {"response": response_text, "preview_updates": self.preview_updates}

    async def _handle_message_with_tools(self, user_message: str) -> str:
        """Handle messages with tool execution support.

        Args:
            user_message: The user's message

        Returns:
            Final AI response after tool execution
        """
        if not self.ai_interface:
            return "Error: AI interface not initialized"

        max_iterations = 10  # Prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            # Get response with potential tool calls
            if iteration == 0:
                # First iteration: send user message
                response_data = await self.ai_interface.get_response_with_tools(user_message)
            else:
                # Subsequent iterations: invoke model with existing conversation history
                response = await self.ai_interface.model.ainvoke(self.ai_interface.conversation_history)
                response_text = response.content if isinstance(response.content, str) else str(response.content)
                tool_calls = getattr(response, "tool_calls", [])

                # Add AI response to history
                self.ai_interface.conversation_history.append(AIMessage(content=response_text, tool_calls=tool_calls))

                response_data = {"text": response_text, "tool_calls": tool_calls}

            # If no tool calls, return the text response
            if not response_data.get("tool_calls"):
                return str(response_data.get("text", ""))

            # Execute tool calls
            tool_results = await self._execute_tool_calls(response_data["tool_calls"])

            # Add tool results to conversation history
            self.ai_interface.conversation_history.extend(tool_results)

            iteration += 1

        return "Maximum tool execution iterations reached."

    async def _execute_tool_calls(self, tool_calls: list) -> list[ToolMessage]:
        """Execute a list of tool calls.

        Args:
            tool_calls: List of tool call dictionaries

        Returns:
            List of ToolMessage objects with results
        """
        tool_results = []

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id", "")

            # Find the tool in current tools
            tool = None
            for current_tool in self.current_tools:
                if current_tool.name == tool_name:
                    tool = current_tool
                    break

            if tool:
                try:
                    # Execute the tool
                    result = await tool._arun(**tool_args)
                    result_str = str(result)

                    # Check if this is a preview update
                    if result_str.startswith("PREVIEW_UPDATE::"):
                        parts = result_str.split("::", 2)
                        if len(parts) == 3:
                            _, preview_title, preview_content = parts
                            # Store preview update for frontend
                            self.preview_updates.append({"title": preview_title, "content": preview_content})
                            # Return success message to AI
                            result_str = f"Successfully updated preview window with title '{preview_title}'"

                    tool_results.append(
                        ToolMessage(
                            content=result_str,
                            tool_call_id=tool_id,
                        )
                    )
                except Exception as e:
                    tool_results.append(
                        ToolMessage(
                            content=f"Error executing tool: {e!s}",
                            tool_call_id=tool_id,
                        )
                    )

        return tool_results

    def export_chat_data(self, provider: str, model: str, temperature: float, messages: list) -> dict:
        """Export chat data for download.

        Args:
            provider: Current provider name
            model: Current model name
            temperature: Current temperature
            messages: List of chat messages

        Returns:
            Dictionary with chat data ready for JSON export
        """
        from datetime import datetime

        chat_data = {
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "messages": messages,
        }

        # Include conversation history
        if self.ai_interface:
            chat_data["conversation_history"] = [
                {
                    "type": msg.__class__.__name__,
                    "content": msg.content if hasattr(msg, "content") else str(msg),
                }
                for msg in self.ai_interface.conversation_history
            ]

        return chat_data

    def restore_chat_data(self, chat_data: dict) -> None:
        """Restore conversation history from chat data.

        Args:
            chat_data: Dictionary with chat data from JSON import
        """
        # Clear existing history
        self.clear_history()

        # Restore conversation history if available
        if "conversation_history" in chat_data and self.ai_interface:
            for hist_msg in chat_data["conversation_history"]:
                if hist_msg["type"] == "HumanMessage":
                    self.ai_interface.conversation_history.append(HumanMessage(content=hist_msg["content"]))
                elif hist_msg["type"] == "AIMessage":
                    self.ai_interface.conversation_history.append(AIMessage(content=hist_msg["content"]))
