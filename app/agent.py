"""
Agentic orchestrator module.

Provides the main AgentRAG class that acts as the primary interface for 
the application, encapsulating memory, vector store, and pipeline logic.
"""

from __future__ import annotations

import logging
from typing import Any, BinaryIO

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.memory import ChatMemory
from app.vector_store import (
    load_vectorstore,
    create_vectorstore,
    delete_vectorstore,
    vectorstore_exists,
    get_document_count,
)
from app.loader import load_and_split
from app.chat import chat

logger = logging.getLogger(__name__)


class AgentRAG:
    """
    Main orchestrator for the Agentic RAG application.
    
    Manages the lifecycle of the vector store and conversational memory,
    providing a clean interface for front-end applications (like Streamlit).
    """

    def __init__(self) -> None:
        """Initialize the agent, loading existing vector store if available."""
        self.memory = ChatMemory()
        self.vectorstore: Chroma | None = load_vectorstore()
        logger.info("AgentRAG initialized (vectorstore ready: %s)", self.is_ready)

    @property
    def is_ready(self) -> bool:
        """Check if the agent has an active vector store with documents."""
        return self.vectorstore is not None

    def get_document_count(self) -> int:
        """Get the number of chunks in the vector store."""
        if not self.is_ready:
            return 0
        return get_document_count(self.vectorstore)  # type: ignore

    def ingest_pdfs(self, files: list[BinaryIO]) -> int:
        """
        Process PDF files and rebuild the vector store.
        
        Args:
            files: List of file-like objects containing PDFs.
            
        Returns:
            Number of chunks created and stored.
            
        Raises:
            ValueError: If no text could be extracted.
        """
        logger.info("Agent instructed to ingest %d PDFs", len(files))
        chunks = load_and_split(files)
        self.vectorstore = create_vectorstore(chunks)
        return len(chunks)

    def ask(self, question: str) -> dict[str, Any]:
        """
        Ask a question to the agent.
        
        Args:
            question: The user's input question.
            
        Returns:
            Dictionary with 'answer', 'retrieval_result', etc.
            
        Raises:
            RuntimeError: If vector store is not initialized.
        """
        if not self.is_ready:
            raise RuntimeError("Cannot ask question: vector store is empty. Please upload documents first.")
            
        return chat(question, self.vectorstore, self.memory)  # type: ignore

    def save_ai_response(self, response: str) -> None:
        """
        Save the final AI response to memory.
        
        This should be called by the frontend after streaming is complete.
        """
        self.memory.add_ai_message(response)

    def clear_memory(self) -> None:
        """Clear the conversational memory."""
        self.memory.clear()
        
    def reset_database(self) -> None:
        """Delete the vector store and reset the agent state."""
        delete_vectorstore()
        self.vectorstore = None
        self.clear_memory()
        logger.info("Agent database and memory reset.")
