"""
Chat interface layer.

Handles high-level chat interactions, managing memory and delegating to the RAG pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_chroma import Chroma

from app.memory import ChatMemory
from app.rag import rag_pipeline

logger = logging.getLogger(__name__)


def chat(
    question: str,
    vectorstore: Chroma,
    memory: ChatMemory,
) -> dict[str, Any]:
    """
    Process a chat message from the user.

    Updates memory with the user's question, calls the RAG pipeline,
    and returns the result. 
    
    NOTE: The caller (e.g. Streamlit app) is responsible for updating the 
    memory with the AI's final response after streaming completes.

    Args:
        question: The user's input question.
        vectorstore: The ChromaDB vector store.
        memory: The ChatMemory instance.

    Returns:
        The dictionary result from the rag_pipeline.
    """
    logger.info("Received chat question.")
    
    # Get history before adding the new question to avoid duplicating it
    history = memory.get_messages()
    
    # Update memory with user question
    memory.add_user_message(question)
    
    # Execute RAG pipeline with streaming enabled
    result = rag_pipeline(
        vectorstore=vectorstore,
        question=question,
        chat_history=history,
        stream=True
    )
    
    return result