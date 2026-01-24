"""Module defining a strategy for handling language models that return dictionary outputs."""

from typing import Generic, cast, override

from langchain_core.language_models import (
    BaseLanguageModel,
    LanguageModelInput,
)
from langchain_core.runnables import Runnable, RunnableLambda

from .base_model_strategy import BaseModelStrategy, ModelTypeVar


class DictModelStrategy(Generic[ModelTypeVar], BaseModelStrategy[ModelTypeVar, dict]):
    """
    Strategy for handling language models that return dictionary outputs.
    """

    @override
    def adapt_llm(self) -> Runnable[LanguageModelInput, dict]:
        if isinstance(self.model, BaseLanguageModel):
            self._model_supports_structured_output = False
            try:
                adapted_model: Runnable[LanguageModelInput, dict] = (  # pyright: ignore[reportAssignmentType]
                    self.model.with_structured_output(
                        self.structured_output_model, include_raw=True
                    )
                )
                self._model_supports_structured_output = True

                def get_raw(result: dict) -> dict:
                    return cast(dict, result.get("raw", {}))

                return adapted_model | RunnableLambda(get_raw) | self.dict_parser
            except NotImplementedError:
                self._model_supports_structured_output = False
        return self.model | self.dict_parser
