"""Ported RAG assistant stack (prompts + LLM routing from ``RAG/``) for rag_service."""

from .assistant import try_llm_assistant_response
from .rag_tools.augment import augment_kb_context

__all__ = ["try_llm_assistant_response", "augment_kb_context"]
