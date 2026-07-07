"""
Text splitting module.

Splits LangChain Document objects into smaller chunks using
RecursiveCharacterTextSplitter while preserving and enriching metadata.
"""

from __future__ import annotations

import logging
from typing import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def split_documents(
    documents: Sequence[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    Split a list of Documents into smaller chunks.

    Each chunk inherits the original document's metadata and receives
    an additional ``chunk_index`` field.

    Args:
        documents: Source documents to split.
        chunk_size: Maximum characters per chunk (default from config).
        chunk_overlap: Overlap between consecutive chunks (default from config).

    Returns:
        List of chunked Document objects with enriched metadata.
    """
    _chunk_size = chunk_size or CHUNK_SIZE
    _chunk_overlap = chunk_overlap or CHUNK_OVERLAP

    logger.info(
        "Splitting %d documents (chunk_size=%d, overlap=%d).",
        len(documents),
        _chunk_size,
        _chunk_overlap,
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_chunk_size,
        chunk_overlap=_chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # Enrich metadata with chunk index per source document
    source_counters: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        idx = source_counters.get(source, 0)
        chunk.metadata["chunk_index"] = idx
        source_counters[source] = idx + 1

    logger.info("Created %d chunks from %d documents.", len(chunks), len(documents))
    return chunks