"""Module defining a strategy for handling language models that return dictionary outputs."""

from typing import Generic, Self, cast, get_origin, override

from langchain_core.language_models import (
    BaseLanguageModel,
    LanguageModelInput,
)
from langchain_core.runnables import Runnable, RunnableLambda

from ..support import ModelTypeVar, BaseFactory
from .base_model_strategy import BaseModelStrategy


class DictModelStrategy(Generic[ModelTypeVar], BaseModelStrategy[dict, ModelTypeVar]):
    """
    Strategy for handling language models that return dictionary outputs.
    """

    @override
    def adapt_llm(self) -> Runnable[LanguageModelInput, dict]:
        if isinstance(self.language_model, BaseLanguageModel):
            self._model_supports_structured_output = False
            try:
                adapted_model: Runnable[LanguageModelInput, dict] = (
                    self.language_model.with_structured_output(
                        self.structured_output_model, include_raw=True
                    )
                )  # pyright: ignore[reportAssignmentType]
                self._model_supports_structured_output = True

                def get_raw(result: dict) -> dict:
                    return cast(dict, result.get("raw", {}))

                return (
                    self.wrap(adapted_model, "Language model with structured output")
                    | self.wrap(
                        RunnableLambda(get_raw, name="Raw output extractor"),
                        "Raw output extractor",
                    )
                    | self.wrap(self.dict_parser, "Dict output parser")
                )
            except NotImplementedError:
                self._model_supports_structured_output = False
        return self.wrap(
            self.language_model, "Language model without structured output"
        ) | self.wrap(self.dict_parser, "Dict output parser")

    @override
    @classmethod
    def should_be_selected(cls, factory: BaseFactory) -> Self | None:
        if (
            factory.structured_model_type is not None
            and get_origin(factory.output_type) is dict
        ):
            return cls(factory)
        return None
