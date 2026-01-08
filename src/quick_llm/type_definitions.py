"""Module defining type aliases for QuickLLM."""

from typing import TypeVar
from langchain_core.language_models import LanguageModelInput
from pydantic import BaseModel


ChainInputType = str | dict | BaseModel
ChainOutputVar = TypeVar("ChainOutputVar")
PromptOutputVar = TypeVar("PromptOutputVar", bound=LanguageModelInput)
