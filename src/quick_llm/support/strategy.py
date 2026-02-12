"""Strategy interface for adapting language models to specific use cases."""

from abc import ABC, abstractmethod
from typing import Generic, Self

from langchain_core.runnables import Runnable

from .base_factory import BaseFactory
from .type_definitions import ChainOutputVar, LanguageModelInput, LanguageModelOutputVar


class Strategy(ABC, Generic[LanguageModelOutputVar]):
    """
    A base abstract strategy interface for adapting language models to specific use cases.

    This class serves as a blueprint for defining strategies that adapt language models
    to different input and output transformations. Each strategy is responsible for:
    - Defining how the model's prompt input is adapted using `prompt_input_adapter`.
    - Specifying how the language model itself is adapted to the strategy via `adapt_llm`.
    - Providing an output transformation mechanism for chaining the model
      output with the desired format.

    Generic Type Parameters:
        LanguageModelOutput: The output type of the language model, which can
        be a string, dictionary, or Pydantic model.
    """

    @property
    @abstractmethod
    def prompt_input_adapter(self) -> Runnable[dict, dict]:
        """Returns a runnable that adapts the chain input to the model prompt input."""

    @property
    @abstractmethod
    def adapted_llm(self) -> Runnable[LanguageModelInput, LanguageModelOutputVar]:
        """Returns the adapted language model runnable."""

    @property
    @abstractmethod
    def output_transformer(self) -> Runnable[LanguageModelOutputVar, ChainOutputVar]:
        """Returns a runnable that transforms the model output to the desired chain output."""

    @classmethod
    def should_be_selected(cls, _factory: BaseFactory) -> Self | None:
        """
        Determines whether this strategy should be selected based on the current context.

        This method can be overridden to provide custom logic for strategy selection,
        such as checking the type of language model, the desired output format, or
        other contextual factors.

        Returns:
            bool: True if this strategy should be selected, False otherwise.
        """
        return None
