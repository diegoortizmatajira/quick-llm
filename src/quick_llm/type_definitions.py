"""Module defining type aliases for QuickLLM."""

from typing import TypeVar
from jsonpatch import Sequence
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import MessageLikeRepresentation
from pydantic import BaseModel


ChainInputType = str | dict | BaseModel | Sequence[MessageLikeRepresentation]
ChainOutputVar = TypeVar("ChainOutputVar")
PromptOutputVar = TypeVar("PromptOutputVar", bound=LanguageModelInput)
