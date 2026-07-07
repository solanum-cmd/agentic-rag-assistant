"""
Prompt templates module.

Contains the system prompts and ChatPromptTemplates used by the agent.
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)


# ---------------------------------------------------------------------------
# Base System Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are DocMind AI, an expert enterprise document assistant.
Your goal is to provide accurate, helpful, and concise answers based 
EXCLUSIVELY on the provided document context.

CRITICAL RULES:
1. ONLY use information from the provided context to answer the question.
2. If the answer cannot be found in the context, clearly state:
   "I couldn't find that information in the uploaded documents."
3. Do NOT hallucinate, guess, or use outside knowledge.
4. If the context contains conflicting information, mention the conflict.
5. Format your answers clearly using Markdown (bullet points, bold text).
6. Always aim to be helpful but strictly grounded in the provided facts.

Below is the retrieved context from the uploaded documents:
----------------------
{context}
----------------------
"""

# ---------------------------------------------------------------------------
# RAG Prompt Template (with Memory)
# ---------------------------------------------------------------------------
# This template expects three variables:
# - context: The retrieved document chunks
# - chat_history: A list of previous BaseMessages
# - question: The user's current question

RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{question}")
])


# ---------------------------------------------------------------------------
# Standalone Question Generator Prompt
# ---------------------------------------------------------------------------
# Used to rephrase a follow-up question into a standalone question 
# that can be used for vector store retrieval.

CONDENSE_QUESTION_SYSTEM_PROMPT = """
Given the following conversation and a follow-up question, rephrase the 
follow-up question to be a standalone question that captures all relevant 
context from the conversation. 

If the follow-up question is already standalone or changes the topic 
completely, just return it as is. Do NOT answer the question, just 
reformulate it.
"""

CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(CONDENSE_QUESTION_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("Follow-up question: {question}")
])