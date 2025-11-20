"""
LLM-based Disease Linking Module

This module provides OpenAI LLM-based disease extraction as an alternative
to scispaCy NER for linking drug indications to MONDO disease IDs.
"""

__version__ = "1.0.0"
__author__ = "Predicate Automate Team"

from .openai_linker import OpenAILinker, LLMMatchResult
from .llm_disease_linker import LLMDiseaseLinkingPipeline, DiseaseMatch, DiseaseValidator

__all__ = [
    "OpenAILinker",
    "LLMMatchResult",
    "LLMDiseaseLinkingPipeline",
    "DiseaseMatch",
    "DiseaseValidator",
]
