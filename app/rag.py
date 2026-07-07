"""
RAG Pipeline module.

Combines retrieval, prompts, memory, and LLM execution into a single cohesive pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_chroma import Chroma
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig

from app.llm import get_llm
from app.prompts import RAG_PROMPT_TEMPLATE, CONDENSE_QUESTION_PROMPT
from app.retriever import retrieve_context, RetrievalResult

logger = logging.getLogger(__name__)


def _condense_question(question: str, chat_history: list[BaseMessage]) -> str:
    """
    Rewrite the question to be standalone if there's chat history.
    """
    if not chat_history:
        return question

    llm = get_llm()
    chain = CONDENSE_QUESTION_PROMPT | llm
    
    try:
        response = chain.invoke({
            "chat_history": chat_history,
            "question": question
        })
        standalone_q = response.content.strip()
        logger.debug("Condense Q: '%s' -> '%s'", question, standalone_q)
        return standalone_q
    except Exception:
        logger.exception("Failed to condense question, falling back to original.")
        return question


def rag_pipeline(
    vectorstore: Chroma,
    question: str,
    chat_history: list[BaseMessage] | None = None,
    stream: bool = True,
) -> dict[str, Any]:
    """
    Execute the full RAG pipeline for a given question.

    Args:
        vectorstore: The ChromaDB vector store.
        question: The user's input question.
        chat_history: Optional list of previous conversation messages.
        stream: If True, returns a streaming iterator for the answer.
                If False, returns the complete answer string.

    Returns:
        A dictionary containing:
        - 'answer': The LLM's response (string or streaming iterator)
        - 'retrieval_result': The RetrievalResult object with context and sources
        - 'standalone_question': The question actually used for search
    """
    history = chat_history or []
    
    # 1. Condense question if history exists
    standalone_q = _condense_question(question, history)
    
    # 2. Retrieve relevant context
    retrieval_result = retrieve_context(vectorstore, standalone_q)
    
    # 3. Construct prompt
    llm = get_llm()
    
    chain = RAG_PROMPT_TEMPLATE | llm
    
    invoke_args = {
        "context": retrieval_result.context,
        "chat_history": history,
        "question": question,
    }
    
    # 4. Generate answer
    try:
        if stream:
            # We return the generator directly so Streamlit can consume it
            answer = chain.stream(invoke_args)
            logger.info("Started streaming LLM response.")
        else:
            response = chain.invoke(invoke_args)
            answer = response.content
            logger.info("Generated LLM response synchronously.")
            
    except Exception:
        logger.exception("LLM generation failed.")
        answer = "I encountered an error while trying to generate a response. Please try again."

    return {
        "answer": answer,
        "retrieval_result": retrieval_result,
        "standalone_question": standalone_q,
    }