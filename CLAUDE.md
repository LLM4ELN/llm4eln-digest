# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

llm4eln-digest is a Python application for Electronic Laboratory Notebook (ELN) digest generation using LLM-powered tools. The project integrates LangChain, LangGraph, and Panel for building an interactive chatbot interface that can interface with AI models (Anthropic Claude, OpenAI GPT) and handle ELN data processing.

## Development Commands

### Environment Setup
```bash
make install              # Create virtual environment, sync dependencies, and install pre-commit hooks
```

### Code Quality
```bash
make check                # Run all quality checks: lock file validation, pre-commit, mypy, and deptry
uv run pre-commit run -a  # Run pre-commit hooks on all files
uv run mypy               # Type checking with mypy (checks src/ directory)
uv run deptry src         # Check for obsolete dependencies
```

### Testing
```bash
make test                 # Run pytest with doctests
uv run python -m pytest   # Run pytest without make
```

### Running the Application
```bash
python -m llm4eln_digest.main  # Start the Panel server on port 5006
```

The application requires a `.env` file with API credentials (see `.env` in `.gitignore`).

### Documentation
```bash
make docs                 # Build and serve MkDocs documentation locally
make docs-test            # Test documentation build without serving
```

### Building and Publishing
```bash
make build                # Build wheel file
make publish              # Publish to PyPI (requires PYPI_TOKEN)
```

## Architecture

### Three-Tier Structure

1. **main.py**: Entry point that loads environment variables and serves the Panel application
2. **frontend.py**: Panel UI layer using Panelini framework with ChatInterface
3. **backend.py**: (Currently empty) Intended for LLM integration and business logic

### Key Components

- **tools/**: Custom tools for LLM agents
  - `pydantic_sparql.py`: SPARQL query handling with Pydantic models

- **utils/**: Shared utilities
  - `ai_interface.py`: Model enums and interfaces for Anthropic (Claude) and Azure OpenAI models
  - `osl_eln.py`: ELN-specific utilities

### Dependencies

Core LLM stack:
- `langchain` and `langchain-community`: LLM framework
- `langchain-anthropic` and `langchain-openai`: Model providers
- `langgraph`: Graph-based agent workflows
- `llm-sandbox[docker]`: Sandboxed code execution

UI:
- `panelini`: Panel wrapper for dashboard apps
- `panel`: Underlying web framework

## Code Style

- Line length: 120 characters
- Type checking enabled with strict mypy configuration
- Ruff for linting and formatting
- Pre-commit hooks enforce code quality
- Tests exempt from S101 (assert statements allowed)

## Python Version Support

Supports Python 3.10 through 3.14. CI/CD tests against all versions.

## File Structure Notes

- Source code in `src/llm4eln_digest/`
- Tests in `tests/` directory
- Uses `hatchling` as build backend
- Lock file (`uv.lock`) is committed for reproducibility
- Local data directories (`data/`, `_ign/`) are gitignored
