"""
Vector store module.

Manages a persistent ChromaDB vector store for document embeddings.
Provides create, load, add, delete, and status-check operations.
"""

from __future__ import annotations

import logging
import shutil
from typing import Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
from app.embeddings import get_embeddings

logger = logging.getLogger(__name__)


def create_vectorstore(documents: Sequence[Document]) -> Chroma:
    """
    Create a new persistent ChromaDB vector store from documents.

    If a store already exists at the persist directory it is **replaced**.

    Args:
        documents: Chunked Document objects to embed and store.

    Returns:
        A Chroma vector store instance.
    """
    logger.info(
        "Creating vector store with %d documents at '%s'.",
        len(documents),
        CHROMA_PERSIST_DIR,
    )

    # Remove existing store to start fresh
    if CHROMA_PERSIST_DIR.exists():
        shutil.rmtree(CHROMA_PERSIST_DIR, ignore_errors=True)
        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Cleared existing vector store.")

    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=list(documents),
        embedding=embeddings,
        persist_directory=str(CHROMA_PERSIST_DIR),
        collection_name=CHROMA_COLLECTION_NAME,
    )

    logger.info("Vector store created successfully (%d documents).", len(documents))
    return vectorstore


def load_vectorstore() -> Chroma | None:
    """
    Load an existing persistent ChromaDB vector store.

    Returns:
        A Chroma vector store instance, or ``None`` if no store exists.
    """
    if not vectorstore_exists():
        logger.info("No existing vector store found at '%s'.", CHROMA_PERSIST_DIR)
        return None

    logger.info("Loading vector store from '%s'.", CHROMA_PERSIST_DIR)
    embeddings = get_embeddings()

    vectorstore = Chroma(
        persist_directory=str(CHROMA_PERSIST_DIR),
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
    )

    count = vectorstore._collection.count()
    logger.info("Vector store loaded with %d documents.", count)
    return vectorstore


def add_documents(
    vectorstore: Chroma,
    documents: Sequence[Document],
) -> Chroma:
    """
    Add new documents to an existing vector store.

    Args:
        vectorstore: Existing Chroma vector store.
        documents: New Document objects to add.

    Returns:
        The updated Chroma vector store.
    """
    logger.info("Adding %d documents to existing vector store.", len(documents))
    vectorstore.add_documents(list(documents))
    new_count = vectorstore._collection.count()
    logger.info("Vector store now contains %d documents.", new_count)
    return vectorstore


def delete_vectorstore() -> None:
    """Delete the persistent vector store from disk."""
    if CHROMA_PERSIST_DIR.exists():
        shutil.rmtree(CHROMA_PERSIST_DIR, ignore_errors=True)
        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Vector store deleted.")
    else:
        logger.info("No vector store to delete.")


def vectorstore_exists() -> bool:
    """
    Check whether a persistent vector store exists and has data.

    Returns:
        True if the store directory exists and contains files.
    """
    if not CHROMA_PERSIST_DIR.exists():
        return False
    # ChromaDB creates a sqlite3 file inside the persist directory
    contents = list(CHROMA_PERSIST_DIR.iterdir())
    return len(contents) > 0


def get_document_count(vectorstore: Chroma) -> int:
    """
    Get the number of documents (chunks) in the vector store.

    Args:
        vectorstore: A Chroma vector store instance.

    Returns:
        Number of stored document chunks.
    """
    try:
        return vectorstore._collection.count()
    except Exception:
        logger.exception("Failed to get document count.")
        return 0
