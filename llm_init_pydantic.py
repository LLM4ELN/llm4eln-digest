"""
Initialize the LLM model from environment variables
"""

import os

from dotenv import load_dotenv
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.azure import AzureProvider

load_dotenv()


def get_llm_model() -> OpenAIResponsesModel:
    """
    Initialize LLM model based on API_PROVIDER from .env

    Supports:
    - Azure OpenAI (API_PROVIDER=azure)
    """
    api_provider = os.getenv("API_PROVIDER", "azure")

    if api_provider == "azure":
        azure_provider = AzureProvider(
            api_key=os.getenv("API_KEY"),
            api_version=os.getenv("API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.getenv("API_ENDPOINT"),
        )

        return OpenAIResponsesModel(os.getenv("API_MODEL", "gpt-4o"), provider=azure_provider)
    else:
        msg = f"Unsupported API_PROVIDER: {api_provider}"
        raise ValueError(msg)


llm = get_llm_model()
