from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from langchain_core.language_models import (
    LanguageModelInput,
    LanguageModelLike,
)
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from quick_llm import ChainOutputVar


def transparent_runner[T]() -> Runnable[T, T]:
    """A runnable that returns the input as output without any changes."""
    return RunnableLambda(lambda x: x)  # type: ignore


LanguageModelOutput = TypeVar("LanguageModelOutput", str, dict, BaseModel)


class BaseStrategy(ABC, Generic[LanguageModelOutput]):
    """
    A base abstract strategy class for adapting language models to specific use cases.

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

    def __init__(self, model: LanguageModelLike):
        self._model: LanguageModelLike = model
        self.__adapted_llm: Runnable[LanguageModelInput, LanguageModelOutput] | None = (
            None
        )

    def prompt_input_adapter(self) -> Runnable[dict, dict]:
        """Returns a runnable that adapts the chain input to the model prompt input."""
        return transparent_runner()

    @abstractmethod
    def adapt_llm(self) -> Runnable[LanguageModelInput, LanguageModelOutput]:
        """
        Returns a runnable that adapts the language model to the strategy.
        """

    @property
    def adapted_llm(self) -> Runnable[LanguageModelInput, LanguageModelOutput]:
        """Returns the adapted language model runnable."""
        if self.__adapted_llm is None:
            self.__adapted_llm = self.adapt_llm()
        return self.__adapted_llm

    def output_transformer(self) -> Runnable[LanguageModelOutput, ChainOutputVar]:
        """Returns a runnable that transforms the model output to the desired chain output."""
        return transparent_runner()
