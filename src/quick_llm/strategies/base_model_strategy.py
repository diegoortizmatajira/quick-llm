"""Base class for model strategies that handle specific model types."""

from typing import Generic, TypeVar, cast
from langchain_core.language_models import LanguageModelLike
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import Runnable
from pydantic import BaseModel
from quick_llm.strategies.base_strategy import BaseStrategy, transparent_runner


ModelTypeVar = TypeVar("ModelTypeVar", BaseModel, dict)
ResultTypeVar = TypeVar("ResultTypeVar", BaseModel, dict)


class BaseModelStrategy(
    Generic[ModelTypeVar, ResultTypeVar], BaseStrategy[ResultTypeVar]
):
    """
    Base class for model strategies that handle specific model types.
    """

    def __init__(
        self, model_type_reference: type[ModelTypeVar], model: LanguageModelLike
    ):
        super().__init__()
        self._model_type_reference = model_type_reference
        self._model_supports_structured_output: bool = False
        self._model: LanguageModelLike = model

    @property
    def dict_parser(self) -> Runnable[dict, dict]:
        """
        Returns a Runnable that parses dictionary outputs from the model.
        """
        # if self._model_supports_structured_output:
        #     return transparent_runner()

        # When structured output is not supported, use JsonOutputParser
        if issubclass(self._model_type_reference, dict):
            # For dict type, use JsonOutputParser without pydantic_object
            return cast(Runnable[dict, dict], JsonOutputParser())

        # For Pydantic models, use JsonOutputParser with pydantic_object
        return cast(
            Runnable[dict, dict],
            JsonOutputParser(pydantic_object=self._model_type_reference),
        )
