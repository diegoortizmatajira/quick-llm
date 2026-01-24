"""Base class for model strategies that handle specific model types."""

from typing import Any, Generic, TypeVar, cast
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from quick_llm import ChainFactory
from quick_llm.type_definitions import ModelTypeVar
from .base_strategy import BaseStrategy


ResultTypeVar = TypeVar("ResultTypeVar", BaseModel, dict, None)


class BaseModelStrategy(
    Generic[ResultTypeVar, ModelTypeVar], BaseStrategy[ResultTypeVar, ModelTypeVar]
):
    """
    Base class for model strategies that handle specific model types.
    """

    def __init__(self, factory: ChainFactory[Any, ModelTypeVar]):
        super().__init__(factory)
        self._model_supports_structured_output: bool = False

    @property
    def structured_output_model(self) -> type[ModelTypeVar]:
        if self.factory.structured_output_model is None:
            raise ValueError("Structured output model is not defined in the factory.")
        # NOTE: This following cast should not be required
        return cast(type[ModelTypeVar], self.factory.structured_output_model)

    @property
    def dict_parser(self) -> Runnable[dict, dict]:
        """
        Returns a Runnable that parses dictionary outputs from the model.
        """
        # if self._model_supports_structured_output:
        #     return transparent_runner()
        # When structured output is not supported, use JsonOutputParser
        if issubclass(self.structured_output_model, dict):
            # For dict type, use JsonOutputParser without pydantic_object
            return cast(Runnable[dict, dict], JsonOutputParser())

        # For Pydantic models, use JsonOutputParser with pydantic_object
        return cast(
            Runnable[dict, dict],
            JsonOutputParser(pydantic_object=self.factory.structured_output_model),
        )
