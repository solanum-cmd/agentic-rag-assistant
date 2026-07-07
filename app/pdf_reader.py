"""
PDF text extraction module.

Uses PyMuPDF (fitz) to extract text from uploaded PDF files,
returning LangChain Document objects with per-page metadata.
"""

from __future__ import annotations

import logging
import re
from io import BytesIO
from typing import BinaryIO

import fitz  # PyMuPDF
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _clean_text(text: str) -> str:
    """
    Clean extracted PDF text by normalising whitespace and removing artefacts.

    Args:
        text: Raw text extracted from a PDF page.

    Returns:
        Cleaned text string.
    """
    # Collapse multiple whitespace / newlines into single spaces
    text = re.sub(r"\s+", " ", text)
    # Remove non-printable characters (keep common unicode)
    text = re.sub(r"[^\x20-\x7E\u00A0-\uFFFF]", "", text)
    return text.strip()


def read_pdf(
    file: BinaryIO,
    filename: str | None = None,
) -> list[Document]:
    """
    Extract text from a PDF file and return a list of LangChain Documents.

    Each Document corresponds to one page and carries metadata including
    the source filename and page number.

    Args:
        file: A file-like object (e.g. Streamlit UploadedFile).
        filename: Optional display name for the source file.

    Returns:
        List of Document objects, one per page with non-empty text.

    Raises:
        ValueError: If the PDF contains no extractable text.
    """
    source_name = filename or getattr(file, "name", "unknown.pdf")
    logger.info("Reading PDF: %s", source_name)

    try:
        raw_bytes = file.read()
        document = fitz.open(stream=raw_bytes, filetype="pdf")
    except Exception:
        logger.exception("Failed to open PDF: %s", source_name)
        raise

    documents: list[Document] = []

    for page_num, page in enumerate(document, start=1):
        raw_text = page.get_text()
        cleaned = _clean_text(raw_text)

        if not cleaned:
            logger.debug("Page %d of '%s' has no text — skipped.", page_num, source_name)
            continue

        doc = Document(
            page_content=cleaned,
            metadata={
                "source": source_name,
                "page": page_num,
                "total_pages": len(document),
            },
        )
        documents.append(doc)

    document.close()

    if not documents:
        msg = f"No extractable text found in '{source_name}'."
        logger.warning(msg)
        raise ValueError(msg)

    logger.info(
        "Extracted %d pages from '%s' (%d total pages).",
        len(documents),
        source_name,
        len(document) if not document.is_closed else documents[-1].metadata["total_pages"],
    )
    return documents


def read_multiple_pdfs(
    files: list[BinaryIO],
) -> list[Document]:
    """
    Extract text from multiple PDF files.

    Args:
        files: List of file-like objects.

    Returns:
        Combined list of Document objects from all PDFs.
    """
    all_documents: list[Document] = []
    for f in files:
        name = getattr(f, "name", "unknown.pdf")
        try:
            docs = read_pdf(f, filename=name)
            all_documents.extend(docs)
        except ValueError as exc:
            logger.warning("Skipping '%s': %s", name, exc)
        except Exception:
            logger.exception("Error processing '%s'", name)
            raise
    logger.info("Total documents extracted from %d PDFs: %d", len(files), len(all_documents))
    return all_documents