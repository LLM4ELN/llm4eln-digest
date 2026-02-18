"""Frontend UI layer for the Chatbot application."""

from typing import Any

import panel as pn
from panelini import Panelini  # type: ignore[import-untyped]

from llm4eln_digest.backend import AiBackend
from llm4eln_digest.tools.basic_tools import AVAILABLE_TOOLS


class Frontend:
    """Frontend class for the Chatbot application."""

    def __init__(self) -> None:
        # Initialize backend
        self.backend = AiBackend()

        # Initialize with get_current_time tool enabled by default
        from llm4eln_digest.tools.basic_tools import get_current_time_tool

        self.backend.update_tools([get_current_time_tool])

        # Initialize preview content with proper overflow handling (starts empty)
        self.preview_content = pn.pane.Markdown(
            "",
            sizing_mode="stretch_both",
            styles={
                "overflow-y": "auto",  # Vertical scroll when content is too long
                "overflow-x": "auto",  # Horizontal scroll when content is too wide
                "max-width": "100%",  # Don't exceed parent width
                "word-wrap": "break-word",  # Break long words
                "overflow-wrap": "break-word",  # Break long words (alternative)
            },
        )

        # Create provider selection widget dynamically from backend
        provider_options = self.backend.get_available_providers()
        self.provider_selector = pn.widgets.Select(
            name="Provider",
            options=provider_options,
            value=self.backend.current_provider,
            sizing_mode="stretch_width",
            margin=(5, 5, 10, 5),
        )

        # Create model selection widget (starts with current provider's models)
        initial_models = self.backend.get_available_models(self.backend.current_provider)
        self.model_selector = pn.widgets.Select(
            name="Model Selection",
            options=initial_models,
            value=self.backend.current_model,
            sizing_mode="stretch_width",
            margin=(5, 5, 10, 5),  # top, right, bottom, left margins
        )

        # Temperature slider
        self.temperature_slider = pn.widgets.FloatSlider(
            name="Temperature",
            start=0.0,
            end=1.0,
            step=0.05,
            value=self.backend.current_temperature,
            sizing_mode="stretch_width",
            margin=(5, 5, 10, 5),  # top, right, bottom, left margins
        )

        # Create tool selection checkboxes
        self.tool_checkboxes: dict[str, dict[str, Any]] = {}
        self.tool_checkbox_group = pn.Column(sizing_mode="stretch_width")

        for tool in AVAILABLE_TOOLS:
            # Enable "get_current_time" tool by default
            default_enabled = tool.name == "get_current_time"
            checkbox = pn.widgets.Checkbox(
                name=tool.name.replace("_", " ").title(),
                value=default_enabled,
                sizing_mode="stretch_width",
                margin=(0, 0, 5, 0),  # Add bottom margin to each checkbox
            )
            checkbox.param.watch(self._on_tool_change, "value")
            self.tool_checkboxes[tool.name] = {"checkbox": checkbox, "tool": tool}
            self.tool_checkbox_group.append(checkbox)

        # Flag to prevent duplicate notifications during provider changes
        self._provider_changing = False

        # Create chat management buttons
        self.clear_chat_button = pn.widgets.Button(
            name="Clear Chat & History",
            button_type="danger",
            sizing_mode="stretch_width",
            margin=(5, 5, 5, 5),
        )
        self.clear_chat_button.on_click(self._on_clear_chat)

        self.download_chat_button = pn.widgets.Button(
            name="Download Chat (JSON)",
            button_type="primary",
            sizing_mode="stretch_width",
            margin=(5, 5, 5, 5),
        )
        self.download_chat_button.on_click(self._on_download_chat)

        self.upload_chat_input = pn.widgets.FileInput(
            accept=".json",
            sizing_mode="stretch_width",
            margin=(5, 5, 5, 5),
        )
        self.upload_chat_input.param.watch(self._on_upload_chat, "value")

        # Display for uploaded filename
        self.uploaded_filename_display = pn.pane.HTML(
            "",
            sizing_mode="stretch_width",
            margin=(0, 5, 5, 5),
        )

        # Watch for changes
        self.provider_selector.param.watch(self._on_provider_change, "value")
        self.model_selector.param.watch(self._on_model_change, "value")
        self.temperature_slider.param.watch(self._on_temperature_change, "value")

        # Create chat interface
        self.chat_interface = pn.chat.ChatInterface(
            callback=self._handle_message,
            user="🧑 User",
            min_width=330,
            show_send=True,
            show_rerun=False,
            show_undo=False,
            show_timestamp=False,
            show_button_name=False,
            show_reaction_icons=False,
            callback_exception="verbose",
            css_classes=["chat-interface"],
            sizing_mode="stretch_both",
        )

        # Create an instance of Panelini
        self.app = Panelini(
            title="LLM4ELN Assistant 🤖",
            sidebar_enabled=True,
            sidebar_visible=False,
        )

        # Add custom CSS to hide the left column and handle preview content overflow
        pn.config.raw_css.append("""
        .bk-panel-models-layout-Column.left {
            display: none !important;
        }

        /* Ensure preview content elements don't overflow */
        .markdown-body pre {
            overflow-x: auto !important;
            max-width: 100% !important;
            white-space: pre-wrap !important;
            word-wrap: break-word !important;
        }

        .markdown-body table {
            display: block !important;
            overflow-x: auto !important;
            max-width: 100% !important;
        }

        .markdown-body img {
            max-width: 100% !important;
            height: auto !important;
        }

        .markdown-body code {
            word-wrap: break-word !important;
            overflow-wrap: break-word !important;
        }
        """)

        # Add general setup to sidebar with nested cards
        self.app.sidebar_set(
            objects=[
                pn.Card(
                    title="🔧 General Setup",
                    collapsible=True,
                    collapsed=False,
                    objects=[
                        pn.Card(
                            title="🌐 Provider Settings",
                            collapsible=True,
                            collapsed=False,
                            objects=[
                                pn.Column(
                                    self.provider_selector,
                                )
                            ],
                            styles={
                                "margin-top": "10px",
                                "margin-bottom": "12px",
                                "padding": "12px",
                            },
                        ),
                        pn.Card(
                            title="⚙️ Model Settings",
                            collapsible=True,
                            collapsed=False,
                            objects=[
                                pn.Column(
                                    self.model_selector,
                                    self.temperature_slider,
                                )
                            ],
                            styles={
                                "margin-bottom": "12px",
                                "padding": "12px",
                            },
                        ),
                        pn.Card(
                            title="🛠️ Basic Tools",
                            collapsible=True,
                            collapsed=False,
                            objects=[
                                pn.Column(
                                    pn.pane.Markdown("**Enable tools for the assistant:**", margin=(0, 0, 10, 0)),
                                    self.tool_checkbox_group,
                                )
                            ],
                            styles={
                                "margin-bottom": "12px",
                                "padding": "12px",
                            },
                        ),
                        pn.Card(
                            title="💬 Chat Management",
                            collapsible=True,
                            collapsed=False,
                            objects=[
                                pn.Column(
                                    pn.pane.Markdown("**Manage conversation:**", margin=(0, 0, 10, 0)),
                                    self.clear_chat_button,
                                    pn.pane.Markdown("**Export/Import:**", margin=(10, 0, 5, 0)),
                                    self.download_chat_button,
                                    pn.pane.Markdown("**Restore from JSON:**", margin=(10, 0, 5, 0)),
                                    self.upload_chat_input,
                                    self.uploaded_filename_display,
                                )
                            ],
                            styles={
                                "margin-bottom": "10px",
                                "padding": "12px",
                            },
                        ),
                    ],
                    styles={"padding": "8px"},
                ),
            ]
        )

        # Add chat card on the left
        chat_card = pn.Card(
            title="💬 Chat",
            collapsible=False,
            objects=[self.chat_interface],
            sizing_mode="stretch_both",
            min_height=600,
            styles={"padding": "15px", "margin-right": "10px"},
        )

        # Add preview card on the right
        preview_card = pn.Card(
            title="📄 Preview",
            collapsible=False,
            objects=[self.preview_content],
            sizing_mode="stretch_both",
            min_height=600,
            styles={
                "padding": "15px",
                "margin-left": "10px",
                "overflow": "hidden",  # Prevent card overflow, let content handle scrolling
            },
        )

        # Create two-column layout using Row with spacing
        # Both cards will take equal width and full height
        main_layout = pn.Row(
            chat_card,
            preview_card,
            sizing_mode="stretch_both",
            min_height=600,
            margin=(10, 15, 10, 15),  # top, right, bottom, left
        )

        # Set the main objects of the Panelini app
        self.app.main_set(objects=[main_layout])

        # Send welcome message from assistant
        self.chat_interface.send(
            value="""Hello! 👋

I'm your ELN (Electronic Lab Notebook) assistant, I can help you with:

- **Documentation** - Recording experiments, observations, and data
- **Organization** - Structuring lab notes and research information
- **Formatting** - Creating clear, well-formatted entries
- **Analysis** - Summarizing findings and results
- **Content Display** - Presenting information in the preview window

How can I help you today?

Feel free to ask me to:

- Create or format lab entries
- Organize experimental data
- Draft protocols or procedures
- Summarize research findings
- Or anything else related to your lab work

What would you like to work on?""",
            user="🤖 Assistant",
            respond=False,
        )

    def _update_preview_content(self, title: str, content: str) -> None:
        """Update the preview window with markdown content.

        Args:
            title: Title for the preview
            content: Markdown content to display
        """
        # Check if content already starts with a heading to avoid duplicates
        if content.strip().startswith("#"):
            # Content already has a heading, use as-is
            self.preview_content.object = content
        else:
            # Add title as heading
            self.preview_content.object = f"# {title}\n\n{content}"

    def _get_selected_tools(self) -> list:
        """Get list of currently selected tools.

        Returns:
            List of enabled tool instances
        """
        selected_tools = []
        for tool_info in self.tool_checkboxes.values():
            if tool_info["checkbox"].value:
                selected_tools.append(tool_info["tool"])
        return selected_tools

    def _on_provider_change(self, event: Any) -> None:
        """Handle provider selection changes."""
        # Set flag to prevent duplicate notifications
        self._provider_changing = True

        # Update model selector options based on new provider
        new_models = self.backend.get_available_models(event.new)
        self.model_selector.options = new_models
        self.model_selector.value = next(iter(new_models.values()))

        # Update backend provider
        provider_display_name, model_name = self.backend.update_provider(event.new)

        # Send single consolidated message
        self.chat_interface.send(
            f"🔄 Switched to **{provider_display_name}** provider with model `{model_name}`. Conversation history cleared.",
            user="⚙️ System",
            respond=False,
        )

        # Reset flag
        self._provider_changing = False

    def _on_model_change(self, event: Any) -> None:
        """Handle model selection changes."""
        # Skip notification if this is part of a provider change
        if self._provider_changing:
            return

        # Update backend model
        model_name = self.backend.update_model(event.new)

        # Notify about model change
        self.chat_interface.send(
            f"🔄 Switched to model `{model_name}`. Conversation history preserved.",
            user="⚙️ System",
            respond=False,
        )

    def _on_temperature_change(self, event: Any) -> None:
        """Handle temperature slider changes."""
        # Update backend temperature
        self.backend.update_temperature(event.new)
        # No notification needed for temperature changes

    def _on_tool_change(self, event: Any) -> None:
        """Handle tool selection changes."""
        _ = event  # Explicitly mark as intentionally unused

        # Update backend tools
        tool_count = self.backend.update_tools(self._get_selected_tools())

        # Notify user about tool changes
        self.chat_interface.send(
            f"🔧 Tools updated. {tool_count} tool(s) now available. Conversation history preserved.",
            user="⚙️ System",
            respond=False,
        )

    def _on_clear_chat(self, event: Any) -> None:
        """Handle clear chat & history button click."""
        _ = event  # Explicitly mark as intentionally unused

        # Clear the conversation history via backend
        self.backend.clear_history()

        # Clear the chat interface
        self.chat_interface.clear()

        # Send notification
        self.chat_interface.send(
            "🗑️ Chat and conversation history cleared.",
            user="⚙️ System",
            respond=False,
        )

    def _on_download_chat(self, event: Any) -> None:
        """Handle download chat button click."""
        import base64
        import json
        from datetime import datetime

        _ = event  # Explicitly mark as intentionally unused

        # Extract messages from chat interface
        messages = []
        for msg in self.chat_interface.objects:
            if hasattr(msg, "object") and hasattr(msg, "user"):
                messages.append({"user": msg.user, "content": str(msg.object)})

        # Get chat data from backend
        provider_name = self.backend.get_provider_display_name(self.backend.current_provider)
        chat_data = self.backend.export_chat_data(
            provider=provider_name,
            model=self.backend.current_model.value,
            temperature=self.backend.current_temperature,
            messages=messages,
        )

        # Create JSON string
        json_str = json.dumps(chat_data, indent=2, ensure_ascii=False)

        # Save to file using Panel's download mechanism
        filename = f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Show download link
        b64 = base64.b64encode(json_str.encode()).decode()
        download_html = f'<a href="data:application/json;base64,{b64}" download="{filename}">💾 Download {filename}</a>'

        self.chat_interface.send(
            download_html,
            user="⚙️ System",
            respond=False,
        )

    def _on_upload_chat(self, event: Any) -> None:
        """Handle chat upload from JSON file."""
        import json

        if not event.new:
            return

        try:
            # Get the filename from the FileInput widget
            filename = self.upload_chat_input.filename if hasattr(self.upload_chat_input, "filename") else "unknown"

            # Decode file content
            content = event.new.decode("utf-8")
            chat_data = json.loads(content)

            # Clear existing chat and history
            self.chat_interface.clear()

            # Restore messages to chat interface
            for msg in chat_data.get("messages", []):
                self.chat_interface.send(msg["content"], user=msg["user"], respond=False)

            # Restore conversation history via backend
            self.backend.restore_chat_data(chat_data)

            # Update filename display with HTML for proper styling
            self.uploaded_filename_display.object = (
                f'<span style="font-size: 0.85em; color: black;">📁 Restored: <code>{filename}</code></span>'
            )

            # Notify success
            self.chat_interface.send(
                f"✅ Chat restored from JSON ({len(chat_data.get('messages', []))} messages).",
                user="⚙️ System",
                respond=False,
            )

            # Clear the file input after successful restore to allow re-uploading the same file
            # Schedule this to happen after the current callback completes
            pn.state.execute(lambda: setattr(self.upload_chat_input, "value", b""))

        except Exception as e:
            self.chat_interface.send(
                f"❌ Error restoring chat: {e!s}",
                user="⚙️ System",
                respond=False,
            )
            # Clear the file input even on error
            pn.state.execute(lambda: setattr(self.upload_chat_input, "value", b""))

    async def _handle_message(self, contents: str, user: str, instance: pn.chat.ChatInterface) -> None:
        """Handle incoming messages and return complete responses.

        Args:
            contents: The user's message
            user: The user identifier (unused but required by Panel)
            instance: The ChatInterface instance (unused but required by Panel)
        """
        _ = user  # Explicitly mark as intentionally unused

        # Check if tools are enabled
        use_tools = len(self._get_selected_tools()) > 0

        # Process message via backend
        result = await self.backend.process_message(contents, use_tools=use_tools)

        # Handle any preview updates
        for preview_update in result.get("preview_updates", []):
            self._update_preview_content(preview_update["title"], preview_update["content"])

        # Send AI response
        instance.send(result["response"], user="🤖 Assistant", respond=False)


if __name__ == "__main__":
    frontend = Frontend()
    pn.serve(frontend.app.servable(), title="Chatbot Panelini Example", port=5006)
