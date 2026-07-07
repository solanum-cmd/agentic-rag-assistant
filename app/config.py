"""
Configuration module for the Agentic RAG Assistant.

Loads environment variables from .env file and provides centralized
configuration with sensible defaults for all components.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
DOCUMENTS_DIR: Path = PROJECT_ROOT / "documents"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
CHROMA_PERSIST_DIR: Path = DATA_DIR / "chroma_db"

# Ensure directories exist
for _dir in (DATA_DIR, DOCUMENTS_DIR, LOGS_DIR, CHROMA_PERSIST_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "google").lower()          # "google" | "openai"
EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()  # "huggingface" | "openai"

# ---------------------------------------------------------------------------
# LLM settings
# ---------------------------------------------------------------------------
LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "")  # auto-selected per provider if blank
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# ---------------------------------------------------------------------------
# Embedding settings
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME: str = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
)

# ---------------------------------------------------------------------------
# Text splitting settings
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

# ---------------------------------------------------------------------------
# Retriever settings
# ---------------------------------------------------------------------------
RETRIEVER_TOP_K: int = int(os.getenv("RETRIEVER_TOP_K", "4"))
RELEVANCE_SCORE_THRESHOLD: float = float(os.getenv("RELEVANCE_SCORE_THRESHOLD", "0.3"))

# ---------------------------------------------------------------------------
# Memory settings
# ---------------------------------------------------------------------------
MEMORY_MAX_TURNS: int = int(os.getenv("MEMORY_MAX_TURNS", "10"))

# ---------------------------------------------------------------------------
# ChromaDB settings
# ---------------------------------------------------------------------------
CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "docmind_collection")

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE: Path = LOGS_DIR / "app.log"


def setup_logging() -> None:
    """Configure application-wide logging with file and console handlers."""
    log_format = "%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # Avoid adding duplicate handlers on reloads
    if root_logger.handlers:
        return

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    root_logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(str(LOG_FILE), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    root_logger.addHandler(file_handler)


def validate_config() -> list[str]:
    """
    Validate critical configuration and return a list of warnings.

    Returns:
        List of warning strings. Empty if everything is configured correctly.
    """
    warnings: list[str] = []

    if LLM_PROVIDER == "google" and not GOOGLE_API_KEY:
        warnings.append("LLM_PROVIDER is 'google' but GOOGLE_API_KEY is not set.")
    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        warnings.append("LLM_PROVIDER is 'openai' but OPENAI_API_KEY is not set.")
    if EMBEDDING_PROVIDER == "openai" and not OPENAI_API_KEY:
        warnings.append("EMBEDDING_PROVIDER is 'openai' but OPENAI_API_KEY is not set.")

    return warnings


# Run logging setup on import
setup_logging()