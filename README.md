# llm4eln-digest

LLM-powered assistant for Electronic Laboratory Notebook (ELN) digest generation.

## Features

- Interactive chat UI served via [Panel](https://panel.holoviz.org/) / [Panelini](https://github.com/opensemanticworld/panelini)
- Multi-provider LLM support: **Anthropic Claude** and **Azure OpenAI**
- Streaming responses with conversation history
- Extensible tool system built on LangChain
- ELN-specific utilities for data processing

## Quick Start

```bash
# 1. Install dependencies
make install

# 2. Create a .env file with your API credentials (see below)

# 3. Run the application
python -m llm4eln_digest.main
```

The UI will be available at `http://localhost:5006`.

## Environment Variables

Create a `.env` file in the project root. Which variables you need depends on the provider you use.

| Variable                    | Provider     | Description                       |
| --------------------------- | ------------ | --------------------------------- |
| `ANTHROPIC_FOUNDRY_API_KEY` | Anthropic    | API key for Anthropic models      |
| `ENDPOINT`                  | Anthropic    | Base URL for the Anthropic API    |
| `AZURE_OPENAI_API_KEY`      | Azure OpenAI | API key for Azure OpenAI          |
| `AZURE_OPENAI_ENDPOINT`     | Azure OpenAI | Endpoint URL vom Azure Deployment |
| `AZURE_OPENAI_API_VERSION`  | Azure OpenAI | API version string                |

## Project Structure

```sh
src/llm4eln_digest/
├── main.py            # Entry point — loads .env, starts Panel server
├── frontend.py        # Chat UI (Panelini ChatInterface)
├── backend.py         # LLM integration and business logic
├── tools/
│   └── basic_tools.py # Custom LangChain tools for agents
└── utils/
    ├── ai_interface.py # Unified interface for Anthropic & Azure OpenAI
    └── osl_eln.py      # ELN-specific utilities
```

## Development

```bash
make check   # Lint, type-check, dependency audit
make test    # Run pytest
make docs    # Build and serve MkDocs locally
```

Requires Python 3.10+. Uses [uv](https://docs.astral.sh/uv/) for dependency management.

## License

[MIT](LICENSE)

## Citation

If you use `llm4eln-digest` in your research, please cite:

```bibtex
@software{llm4eln-digest-2026,
  title = {llm4eln-digest: Summarize, extract, and analyze ELN data utilizing LLMs.},
  author = {LLM4ELN Contributors},
  year = {2026},
  url = {https://github.com/LLM4ELN/llm4eln-digest}
}
