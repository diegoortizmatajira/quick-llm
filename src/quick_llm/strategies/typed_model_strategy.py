"""TypedModelStrategy: A strategy for adapting language models to produce typed outputs."""

from dataclasses import is_dataclass
from typing import Generic, Self, cast, override

from langchain_core.language_models import (
    BaseChatModel,
    LanguageModelInput,
)
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from ..support import ModelTypeVar, BaseFactory
from .base_model_strategy import BaseModelStrategy
from .base_strategy import transparent_runner


class TypedModelStrategy(
    Generic[ModelTypeVar], BaseModelStrategy[ModelTypeVar, ModelTypeVar]
):
    """
    TypedModelStrategy is a generic strategy that adapts a language model to produce
    typed outputs that match the provided type reference.

    This class is responsible for creating an adaptable language model pipeline capable
    of converting raw inputs or model outputs into structured, type-safe results. It extends
    the BaseModelStrategy and is parameterized with a specific model type.

    Attributes:
        _model_type_reference: The type reference for the model.
        _model: The language model that this strategy will adapt.

    Methods:
        __init__(type_reference, model):
            Initializes the strategy with the specified type reference and language model.
        adapt_llm():
            Adapts the language model into a runnable pipeline for producing typed outputs.
        typed_parser():
            Returns a Runnable that parses dictionaries into the specified typed model.
    """

    @override
    def adapt_llm(self) -> Runnable[LanguageModelInput, ModelTypeVar]:
        """
        Adapts the language model to produce typed outputs.

        This method configures the language model pipeline to generate structured,
        type-safe results. If the model supports structured output for the given
        type reference and is a subclass of BaseChatModel, it uses this capability.
        Otherwise, it falls back to using the dictionary-to-typed-model parsing approach.

        Returns:
            Runnable[LanguageModelInput, ModelTypeVar]: A runnable pipeline that
            processes language model inputs and generates typed outputs.
        """
        # Don't use with_structured_output for dict type as it returns AIMessage
        if isinstance(self.language_model, BaseChatModel) and not issubclass(
            self.structured_output_model, dict
        ):
            self._model_supports_structured_output = False
            try:
                model = self.language_model.with_structured_output(
                    self.structured_output_model
                )
                self._model_supports_structured_output = True
                return cast(Runnable[LanguageModelInput, ModelTypeVar], model)
            except NotImplementedError:
                self._model_supports_structured_output = False

        return self.language_model | self.dict_parser | self.typed_parser

    @property
    def typed_parser(self) -> Runnable[dict, ModelTypeVar]:
        """
        Returns a Runnable that parses dictionaries into the specified typed model.

        If the model type reference is a dictionary, a transparent runner is returned.
        Otherwise, it defines a parsing function that validates and converts a dictionary
        into the specified Pydantic model.

        Returns:
            Runnable[dict, ModelTypeVar]: A runnable parser that converts a dictionary
            into a structured typed model.
        """
        if issubclass(self.structured_output_model, dict):
            return transparent_runner()

        def parse_typed(result: dict) -> ModelTypeVar:
            if issubclass(self.structured_output_model, BaseModel):
                return cast(
                    ModelTypeVar, self.structured_output_model.model_validate(result)
                )
            if is_dataclass(self.structured_output_model):
                return cast(
                    ModelTypeVar,
                    self.structured_output_model(**result),  # type: ignore[call-arg]
                )
            # Fallback: assume TypedDict
            return cast(ModelTypeVar, result)

        return RunnableLambda(parse_typed, name="Typed Parser")

    @override
    @classmethod
    def should_be_selected(cls, factory: BaseFactory) -> Self | None:
        if (
            factory.structured_model_type is not None
            and factory.output_type is factory.structured_model_type
        ):
            return cls(factory)
        return None
