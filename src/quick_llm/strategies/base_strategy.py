from abc import ABC, abstractmethod
from typing import Any, Generic, override

from langchain_core.language_models import (
    LanguageModelInput,
    LanguageModelLike,
)
from langchain_core.runnables import Runnable, RunnableLambda

from ..factory.base_factory import BaseFactory
from ..support import (
    ModelTypeVar,
    ChainOutputVar,
    LanguageModelOutputVar,
    Strategy,
)


def transparent_runner[T]() -> Runnable[T, T]:
    """A runnable that returns the input as output without any changes."""
    return RunnableLambda(lambda x: x)  # type: ignore


class BaseStrategy(
    Generic[LanguageModelOutputVar, ModelTypeVar], Strategy[LanguageModelOutputVar], ABC
):
    """
    A base abstract strategy class fordapting language models to specific use cases.

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

    def __init__(self, factory: BaseFactory[Any, ModelTypeVar]):
        self.__factory = factory
        self.__adapted_llm: (
            Runnable[LanguageModelInput, LanguageModelOutputVar] | None
        ) = None

    @property
    def factory(self) -> BaseFactory[Any, ModelTypeVar]:
        """Returns the chain factory associated with the strategy."""
        return self.__factory

    @property
    def language_model(self) -> LanguageModelLike:
        """Returns the language model from the factory."""
        return self.__factory.language_model

    @override
    def prompt_input_adapter(self) -> Runnable[dict, dict]:
        """Returns a runnable that adapts the chain input to the model prompt input."""
        return transparent_runner()

    @abstractmethod
    def adapt_llm(self) -> Runnable[LanguageModelInput, LanguageModelOutputVar]:
        """
        Returns a runnable that adapts the language model to the strategy.
        """

    @property
    @override
    def adapted_llm(self) -> Runnable[LanguageModelInput, LanguageModelOutputVar]:
        """Returns the adapted language model runnable."""
        if self.__adapted_llm is None:
            self.__adapted_llm = self.adapt_llm()
        return self.__adapted_llm

    @override
    def output_transformer(self) -> Runnable[LanguageModelOutputVar, ChainOutputVar]:
        """Returns a runnable that transforms the model output to the desired chain output."""
        return transparent_runner()

    def wrap[Input, Output](
        self, runnable: Runnable[Input, Output], caption: str
    ) -> Runnable[Input, Output]:
        """
        Wraps a runnable with a caption for better traceability.
        """
        return self.factory.wrap(runnable, caption)
