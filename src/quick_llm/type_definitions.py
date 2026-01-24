"""Module defining type aliases for QuickLLM."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from jsonpatch import Sequence
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import MessageLikeRepresentation
from langchain_core.runnables import Runnable
from pydantic import BaseModel


ChainInputType = str | dict | BaseModel | Sequence[MessageLikeRepresentation]
ChainOutputVar = TypeVar("ChainOutputVar")
PromptOutputVar = TypeVar("PromptOutputVar", bound=LanguageModelInput)
LanguageModelOutputVar = TypeVar("LanguageModelOutputVar", str, dict, BaseModel, None)
ModelTypeVar = TypeVar("ModelTypeVar", str, dict, BaseModel, None)


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

    @abstractmethod
    def prompt_input_adapter(self) -> Runnable[dict, dict]:
        """Returns a runnable that adapts the chain input to the model prompt input."""

    @property
    @abstractmethod
    def adapted_llm(self) -> Runnable[LanguageModelInput, LanguageModelOutputVar]:
        """Returns the adapted language model runnable."""

    @abstractmethod
    def output_transformer(self) -> Runnable[LanguageModelOutputVar, ChainOutputVar]:
        """Returns a runnable that transforms the model output to the desired chain output."""
