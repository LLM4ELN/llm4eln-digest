# llm4eln-digest

[![Release](https://img.shields.io/github/v/release/LLM4ELN/llm4eln-digest)](https://img.shields.io/github/v/release/LLM4ELN/llm4eln-digest)
[![Build status](https://img.shields.io/github/actions/workflow/status/LLM4ELN/llm4eln-digest/main.yml?branch=main)](https://github.com/LLM4ELN/llm4eln-digest/actions/workflows/main.yml?query=branch%3Amain)
[![Commit activity](https://img.shields.io/github/commit-activity/m/LLM4ELN/llm4eln-digest)](https://img.shields.io/github/commit-activity/m/LLM4ELN/llm4eln-digest)
[![License](https://img.shields.io/github/license/LLM4ELN/llm4eln-digest)](https://img.shields.io/github/license/LLM4ELN/llm4eln-digest)

An LLM-powered application for Electronic Laboratory Notebook (ELN) digest generation. It provides an interactive chat interface for working with ELN data using AI models from Anthropic and Azure OpenAI.

## Features

- **Multi-provider AI support** — Switch between Anthropic (Claude) and Azure OpenAI (GPT) models at runtime via a sidebar selector.
- **Interactive chat interface** — Built with [Panel](https://panel.holoviz.org/) and [Panelini](https://github.com/LLM4ELN/panelini), featuring streaming responses, tool execution, and a live preview pane.
- **Configurable tools** — Enable or disable built-in tools (current time, metadata statistics, preview updates) from the sidebar.
- **Chat management** — Clear, export (JSON), and restore conversation history.
- **YAML-driven configuration** — Providers, models, and credentials are defined in `config.yml` with `${ENV_VAR}` substitution from `.env`.

## Architecture

```text
main.py          → Entry point: loads .env, serves the Panel app
frontend.py      → UI layer: chat interface, sidebar controls, preview pane
backend.py       → Business logic: AI interface management, tool execution, history
utils/
  config.py      → YAML config loader (ModelConfig, ProviderConfig, AppConfig)
  ai_interface.py→ Unified LLM interface with provider registry and streaming
  osl_eln.py     → ELN-specific utilities (placeholder)
tools/
  basic_tools.py → LangChain tools: get_current_time, calculate_metadata_stats, update_preview
```

## Quick start

```bash
# Install dependencies
make install

# Create .env with your API credentials
cp .env.example .env  # then edit with your keys

# Run the application
python -m llm4eln_digest.main
```

The application starts a Panel server on [http://localhost:5006](http://localhost:5006).

## Development

```bash
make check    # Run all quality checks (pre-commit, mypy, deptry)
make test     # Run pytest with doctests
make docs     # Build and serve MkDocs documentation locally
```
