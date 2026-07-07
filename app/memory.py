"""
Conversational memory module.

Manages the chat history for the agent, maintaining context across turns.
"""

from __future__ import annotations

import logging
from typing import Sequence

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.chat_history import InMemoryChatMessageHistory

from app.config import MEMORY_MAX_TURNS

logger = logging.getLogger(__name__)


class ChatMemory:
    """
    Manages conversational memory for the RAG assistant.
    
    Stores human and AI messages and provides methods to retrieve formatted history.
    """

    def __init__(self, max_turns: int | None = None) -> None:
        """
        Initialize the chat memory.
        
        Args:
            max_turns: Maximum number of conversation turns (question + answer) 
                       to keep in memory. Older turns are discarded.
        """
        self.history = InMemoryChatMessageHistory()
        self.max_turns = max_turns or MEMORY_MAX_TURNS
        logger.info("Initialized ChatMemory (max_turns=%d)", self.max_turns)

    def add_user_message(self, message: str) -> None:
        """Add a user message to the history."""
        self.history.add_user_message(message)
        logger.debug("Added user message to memory.")

    def add_ai_message(self, message: str) -> None:
        """Add an AI response to the history."""
        self.history.add_ai_message(message)
        logger.debug("Added AI message to memory.")
        self._prune_history()

    def get_messages(self) -> list[BaseMessage]:
        """Get all messages currently in memory."""
        return self.history.messages

    def get_formatted_history(self) -> str:
        """
        Get the conversation history formatted as a string for inclusion in prompts.
        
        Returns:
            Formatted string of previous turns, or empty string if history is empty.
        """
        messages = self.history.messages
        if not messages:
            return ""

        formatted_turns = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted_turns.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted_turns.append(f"Assistant: {msg.content}")
                
        return "\n".join(formatted_turns)

    def clear(self) -> None:
        """Clear the conversation history."""
        self.history.clear()
        logger.info("Cleared chat memory.")

    def _prune_history(self) -> None:
        """Keep only the most recent max_turns in memory."""
        messages = self.history.messages
        # A "turn" is typically a User + Assistant pair, so 2 messages
        max_messages = self.max_turns * 2
        
        if len(messages) > max_messages:
            # Keep the last max_messages
            pruned_messages = messages[-max_messages:]
            self.history.clear()
            self.history.add_messages(pruned_messages)
            logger.debug("Pruned memory to last %d messages.", max_messages)

    @property
    def is_empty(self) -> bool:
        """Check if the memory is empty."""
        return len(self.history.messages) == 0
