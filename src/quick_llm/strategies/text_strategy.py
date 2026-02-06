"""A strategy module for handling string-based outputs from language models."""

from typing import Any, Self, override

from langchain_core.language_models import (
    LanguageModelInput,
    LanguageModelOutput,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from ..support import BaseFactory
from .base_strategy import BaseStrategy


class TextStrategy(BaseStrategy[Any, Any]):
    """A strategy for handling and adapting text-based outputs.

    This class extends the BaseStrategy specifically for string
    outputs and provides mechanisms to adapt a language model
    to handle these outputs as well as parse them.
    """

    @override
    def adapt_llm(self) -> Runnable[LanguageModelInput, str]:
        """Adapts the language model to produce string-based outputs.

        This method chains the language model with a text parser to generate
        outputs that are specifically in string format.

        Returns:
            Runnable: A runnable instance that processes LanguageModelInput
            to produce a string output.
        """
        return self.wrap(self.language_model, "Language Model") | self.wrap(
            self.text_parser, "Text Parser"
        )

    @property
    def text_parser(self) -> Runnable[LanguageModelOutput, str]:
        """Provides a text output parser for the language model.

        This property returns a Runnable instance that processes the raw language
        model output into a string format using the StrOutputParser.

        Returns:
            Runnable: An instance of StrOutputParser to convert LanguageModelOutput
            into a string output.
        """
        return StrOutputParser()

    @override
    @classmethod
    def should_be_selected(cls, factory: BaseFactory) -> Self | None:
        if factory.output_type is str:
            return cls(factory)
        return None
