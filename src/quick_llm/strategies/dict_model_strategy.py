"""Module defining a strategy for handling language models that return dictionary outputs."""

from typing import Generic, cast, override
from langchain_core.language_models import (
    BaseChatModel,
    LanguageModelInput,
    LanguageModelLike,
)
from langchain_core.runnables import Runnable, RunnableLambda
from quick_llm.strategies.base_model_strategy import BaseModelStrategy, ModelTypeVar


class DictModelStrategy(Generic[ModelTypeVar], BaseModelStrategy[ModelTypeVar, dict]):
    """
    Strategy for handling language models that return dictionary outputs.
    """

    def __init__(self, type_reference: type[ModelTypeVar], model: LanguageModelLike):
        super().__init__(type_reference, model)

    @override
    def adapt_llm(self) -> Runnable[LanguageModelInput, dict]:
        if isinstance(self._model, BaseChatModel):
            try:

                def get_raw(result: dict) -> dict:
                    return cast(dict, result.get("raw", {}))

                model = self._model.with_structured_output(
                    self._model_type_reference, include_raw=True
                )
                return model | RunnableLambda(get_raw)
            except NotImplementedError:
                self._model_supports_structured_output = False
        return self._model | self.dict_parser
