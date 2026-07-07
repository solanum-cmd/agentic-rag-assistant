"""
LLM module.

Provides a factory function that returns the configured LLM model.
Supports Google Gemini and OpenAI providers with streaming enabled.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import (
    LLM_PROVIDER,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    GOOGLE_API_KEY,
    OPENAI_API_KEY,
)

logger = logging.getLogger(__name__)

# Cache the LLM instance so it's only loaded once
_llm_instance: BaseChatModel | None = None


def get_llm() -> BaseChatModel:
    """
    Return the configured LLM model (singleton).

    The provider is selected via the ``LLM_PROVIDER`` config variable:
    - ``"google"``: Uses Google Gemini models (e.g., gemini-2.5-flash).
    - ``"openai"``: Uses OpenAI models (e.g., gpt-4o-mini).

    Returns:
        A LangChain-compatible BaseChatModel instance.

    Raises:
        ValueError: If the provider is not recognised.
    """
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    if LLM_PROVIDER == "google":
        model_name = LLM_MODEL_NAME or "gemini-2.5-flash"
        logger.info("Loading Google LLM: %s", model_name)
        from langchain_google_genai import ChatGoogleGenerativeAI

        _llm_instance = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=GOOGLE_API_KEY,
            temperature=LLM_TEMPERATURE,
            max_output_tokens=LLM_MAX_TOKENS,
            streaming=True,
        )

    elif LLM_PROVIDER == "openai":
        model_name = LLM_MODEL_NAME or "gpt-4o-mini"
        logger.info("Loading OpenAI LLM: %s", model_name)
        from langchain_openai import ChatOpenAI

        _llm_instance = ChatOpenAI(
            model=model_name,
            api_key=OPENAI_API_KEY,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            streaming=True,
        )

    else:
        msg = f"Unknown LLM_PROVIDER: '{LLM_PROVIDER}'. Use 'google' or 'openai'."
        logger.error(msg)
        raise ValueError(msg)

    logger.info("LLM loaded successfully (provider=%s).", LLM_PROVIDER)
    return _llm_instance


def reset_llm() -> None:
    """Clear the cached LLM instance (useful for testing or provider switch)."""
    global _llm_instance
    _llm_instance = None
    logger.info("LLM instance cache cleared.")