"""A strategy for not modifying the outputs from language models."""

from typing import Generic, cast, override

from langchain_core.language_models import (
    LanguageModelInput,
)
from langchain_core.runnables import Runnable

from quick_llm import LanguageModelOutputVar
from .base_strategy import BaseStrategy


class NullStrategy(
    Generic[LanguageModelOutputVar],
    BaseStrategy[LanguageModelOutputVar, LanguageModelOutputVar],
):
    """A strategy for handling and adapting text-based outputs.

    This class extends the BaseStrategy specifically for string
    outputs and provides mechanisms to adapt a language model
    to handle these outputs as well as parse them.
    """

    @override
    def adapt_llm(self) -> Runnable[LanguageModelInput, LanguageModelOutputVar]:
        return cast(Runnable[LanguageModelInput, LanguageModelOutputVar], self.language_model)
