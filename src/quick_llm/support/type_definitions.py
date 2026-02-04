"""Module defining type aliases for QuickLLM."""

from typing import TypeVar
from jsonpatch import Sequence
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import MessageLikeRepresentation
from pydantic import BaseModel


ChainInputType = str | dict | BaseModel | Sequence[MessageLikeRepresentation]
ChainOutputVar = TypeVar("ChainOutputVar")
PromptOutputVar = TypeVar("PromptOutputVar", bound=LanguageModelInput)
LanguageModelOutputVar = TypeVar("LanguageModelOutputVar", str, dict, BaseModel, None)
ModelTypeVar = TypeVar("ModelTypeVar", str, dict, BaseModel, None)


def is_typed_dict(structured_type: type) -> bool:
    """Check if the given structured type is a TypedDict."""
    return isinstance(structured_type, type) and hasattr(structured_type, "__total__")
