"""
Embeddings module.

Provides a factory function that returns the configured embedding model.
Supports HuggingFace (local, free) and OpenAI providers.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.embeddings import Embeddings

from app.config import EMBEDDING_PROVIDER, EMBEDDING_MODEL_NAME, OPENAI_API_KEY

logger = logging.getLogger(__name__)

# Cache the embedding model instance so it's only loaded once
_embedding_instance: Embeddings | None = None


def get_embeddings() -> Embeddings:
    """
    Return the configured embedding model (singleton).

    The provider is selected via the ``EMBEDDING_PROVIDER`` config variable:
    - ``"huggingface"`` (default): Uses ``sentence-transformers/all-MiniLM-L6-v2``
      locally — no API key required.
    - ``"openai"``: Uses ``text-embedding-3-small`` via the OpenAI API.

    Returns:
        A LangChain-compatible Embeddings instance.

    Raises:
        ValueError: If the provider is not recognised.
    """
    global _embedding_instance
    if _embedding_instance is not None:
        return _embedding_instance

    if EMBEDDING_PROVIDER == "huggingface":
        logger.info("Loading HuggingFace embeddings: %s", EMBEDDING_MODEL_NAME)
        from langchain_huggingface import HuggingFaceEmbeddings

        _embedding_instance = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    elif EMBEDDING_PROVIDER == "openai":
        model = EMBEDDING_MODEL_NAME if "embedding" in EMBEDDING_MODEL_NAME else "text-embedding-3-small"
        logger.info("Loading OpenAI embeddings: %s", model)
        from langchain_openai import OpenAIEmbeddings

        _embedding_instance = OpenAIEmbeddings(
            model=model,
            openai_api_key=OPENAI_API_KEY,
        )

    else:
        msg = f"Unknown EMBEDDING_PROVIDER: '{EMBEDDING_PROVIDER}'. Use 'huggingface' or 'openai'."
        logger.error(msg)
        raise ValueError(msg)

    logger.info("Embeddings loaded successfully (provider=%s).", EMBEDDING_PROVIDER)
    return _embedding_instance


def reset_embeddings() -> None:
    """Clear the cached embedding instance (useful for testing or provider switch)."""
    global _embedding_instance
    _embedding_instance = None
    logger.info("Embedding instance cache cleared.")
