from dotenv import load_dotenv
from panel import serve

from llm4eln_digest.frontend import Frontend


def main() -> None:
    """Entry point for the llm4eln-digest application."""
    load_dotenv()
    ui = Frontend()
    serve(
        ui.app.servable(),
        title="LLM4ELN Assistant 🤖",
        port=5006,
    )


if __name__ == "__main__":
    main()
