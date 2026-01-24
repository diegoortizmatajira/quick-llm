"""Strategy module for Quick LLM."""

from .typed_model_strategy import TypedModelStrategy
from .dict_model_strategy import DictModelStrategy
from .text_strategy import TextStrategy
from .null_strategy import NullStrategy

__all__ = ["TypedModelStrategy", "DictModelStrategy", "TextStrategy", "NullStrategy"]
