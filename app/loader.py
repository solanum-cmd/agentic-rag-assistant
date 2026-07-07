"""
Document loader orchestrator.

Provides a single entry-point that accepts uploaded files, extracts text
via pdf_reader, splits into chunks via text_splitter, and returns
Documents ready for embedding and vector-store ingestion.
"""

from __future__ import annotations

import logging
from typing import BinaryIO, Sequence

from langchain_core.documents import Document

from app.pdf_reader import read_pdf, read_multiple_pdfs
from app.text_splitter import split_documents

logger = logging.getLogger(__name__)


def load_and_split(
    files: Sequence[BinaryIO],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    Full ingest pipeline: extract text from PDFs → split into chunks.

    Args:
        files: One or more file-like objects (e.g. Streamlit UploadedFiles).
        chunk_size: Optional override for chunk size.
        chunk_overlap: Optional override for chunk overlap.

    Returns:
        List of chunked Document objects ready for embedding.

    Raises:
        ValueError: If no extractable text is found in any uploaded file.
    """
    logger.info("Starting document ingest pipeline for %d file(s).", len(files))

    # Step 1: Extract text from PDFs
    raw_documents = read_multiple_pdfs(list(files))

    if not raw_documents:
        msg = "No text could be extracted from the uploaded files."
        logger.error(msg)
        raise ValueError(msg)

    logger.info("Extracted %d raw document pages.", len(raw_documents))

    # Step 2: Split into chunks
    chunks = split_documents(
        raw_documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    logger.info(
        "Ingest pipeline complete: %d files → %d pages → %d chunks.",
        len(files),
        len(raw_documents),
        len(chunks),
    )
    return chunks
