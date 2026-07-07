"""
Retrieval module.

Performs similarity search against the ChromaDB vector store and returns
relevant document chunks with relevance scores and formatted context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import RETRIEVER_TOP_K, RELEVANCE_SCORE_THRESHOLD

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Container for retrieval output."""

    context: str
    """Formatted context string for the LLM prompt."""

    source_documents: list[Document] = field(default_factory=list)
    """Retrieved source Document objects."""

    scores: list[float] = field(default_factory=list)
    """Relevance scores for each retrieved document (lower = more similar for L2)."""


def retrieve_context(
    vectorstore: Chroma,
    question: str,
    top_k: int | None = None,
    score_threshold: float | None = None,
) -> RetrievalResult:
    """
    Retrieve relevant document chunks for a question.

    Uses ChromaDB's similarity search with relevance scores to find
    the most relevant chunks, filters by score threshold, and builds
    a formatted context string with source attributions.

    Args:
        vectorstore: The Chroma vector store to search.
        question: The user's question.
        top_k: Number of chunks to retrieve (default from config).
        score_threshold: Minimum relevance score to include (default from config).

    Returns:
        A RetrievalResult containing the context string, source documents,
        and relevance scores.
    """
    k = top_k or RETRIEVER_TOP_K
    threshold = score_threshold or RELEVANCE_SCORE_THRESHOLD

    logger.info("Retrieving top-%d chunks for question: '%.80s...'", k, question)

    try:
        results = vectorstore.similarity_search_with_relevance_scores(
            query=question,
            k=k,
        )
    except Exception:
        logger.exception("Similarity search failed.")
        return RetrievalResult(context="", source_documents=[], scores=[])

    if not results:
        logger.warning("No results returned from similarity search.")
        return RetrievalResult(context="", source_documents=[], scores=[])

    # Filter by relevance score (higher = more relevant for cosine)
    filtered: list[tuple[Document, float]] = []
    for doc, score in results:
        if score >= threshold:
            filtered.append((doc, score))
        else:
            logger.debug(
                "Filtered out chunk (score=%.3f < threshold=%.3f): '%.50s...'",
                score,
                threshold,
                doc.page_content,
            )

    if not filtered:
        logger.warning(
            "All %d results below relevance threshold (%.3f).",
            len(results),
            threshold,
        )
        # Fall back to top result even if below threshold
        filtered = [(results[0][0], results[0][1])]

    # Build context string with source attributions
    context_parts: list[str] = []
    source_documents: list[Document] = []
    scores: list[float] = []

    for i, (doc, score) in enumerate(filtered, start=1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        context_parts.append(
            f"[Source {i}: {source}, Page {page}]\n{doc.page_content}"
        )
        source_documents.append(doc)
        scores.append(round(score, 4))

    context = "\n\n---\n\n".join(context_parts)

    logger.info(
        "Retrieved %d relevant chunks (scores: %s).",
        len(source_documents),
        scores,
    )

    return RetrievalResult(
        context=context,
        source_documents=source_documents,
        scores=scores,
    )