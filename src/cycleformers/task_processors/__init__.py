"""Processors module for transforming various dataset formats into cycleformers-compatible format.

This module provides a flexible API for converting different dataset formats into cycleformers-compatible format.
Each processor handles a specific dataset format or task type (e.g., NER, machine translation, etc.) while sharing
common functionality through the base class.
"""

from .base import BaseProcessor, ProcessorConfig
from .ner import CONLL2003Processor, CONLL2003ProcessorConfig
from .translation import TranslationProcessor, TranslationProcessorConfig


__all__ = [
    "BaseProcessor",
    "CONLL2003Processor",
    "CONLL2003ProcessorConfig",
    "ProcessorConfig",
    "TranslationProcessor",
    "TranslationProcessorConfig",
]
