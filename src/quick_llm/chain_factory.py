"""Factory class for managing language model instances."""

import logging
from typing import Callable, Generic, Self

from langchain_core.language_models import (
    BaseLanguageModel,
)
from langchain_core.language_models.base import LanguageModelOutputVar
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.runnables import RunnableGenerator
from langchain_core.runnables.base import RunnableSerializable

from quick_llm.prompt_input_parser import PromptInputParser
from quick_llm.type_definitions import ChainInputType, ChainOutputVar, PromptOutputVar


class ChainFactory(
    Generic[
        PromptOutputVar,
        LanguageModelOutputVar,
        ChainOutputVar,
    ]
):
    """Factory class for managing language model instances."""

    def __init__(self):
        self.__logger = logging.getLogger(__name__)
        self.__input_transformer: RunnableGenerator[ChainInputType, dict] | None = None
        self.__output_transformer: (
            RunnableGenerator[LanguageModelOutputVar, ChainOutputVar] | None
        ) = None
        self.__language_model: BaseLanguageModel[LanguageModelOutputVar] | None = None
        self.__prompt_template: BasePromptTemplate[PromptOutputVar] | None = None
        self.__param_input: str = "input"
        self.__param_format_instructions: str = "format_instructions"

    def __fail(self, message: str) -> Exception:
        self.__logger.error(message)
        return ValueError(message)

    @property
    def language_model(self) -> BaseLanguageModel[LanguageModelOutputVar]:
        """
        Gets the language model instance.

        :return: The current instance of BaseLanguageModel or None if not set.
        """
        if self.__language_model is None:
            raise self.__fail("Language model is not set.")
        return self.__language_model

    @property
    def prompt_template(self) -> BasePromptTemplate[PromptOutputVar]:
        """
        Gets the prompt template instance.

        :return: The current instance of PromptTemplate or None if not set.
        """
        if self.__prompt_template is None:
            raise self.__fail("Prompt template is not set.")
        return self.__prompt_template

    @property
    def input_transformer(self) -> RunnableGenerator[ChainInputType, dict]:
        """
        Gets the input transformer instance.

        :return: The current instance of RunnableGenerator for input transformation.
        """
        if self.__input_transformer is None:
            self.__input_transformer = PromptInputParser(self.__param_input)
        return self.__input_transformer

    @property
    def output_transformer(
        self,
    ) -> RunnableGenerator[LanguageModelOutputVar, ChainOutputVar]:
        """
        Gets the output transformer instance.

        :return: The current instance of RunnableGenerator for output transformation.
        """
        if self.__output_transformer is None:
            raise self.__fail("Output transformer is not set.")
        return self.__output_transformer

    def use(self, visitor: Callable[[Self], None]) -> Self:
        """
        Applies a visitor function to the ChainFactory instance.

        :param visitor: A callable that takes a ChainFactory instance and returns None.
        :return: The ChainFactory instance for method chaining.
        """
        self.__logger.debug("Applying visitor to ChainFactory")
        visitor(self)
        return self

    def use_language_model(self, language_model: BaseLanguageModel) -> Self:
        """
        Sets the language model instance.

        :param language_model: An instance of BaseLanguageModel to set.
        :return: The ChainFactory instance for method chaining.
        """
        self.__language_model = language_model
        self.__logger.debug("Setting language model: %s", self.__language_model)
        return self

    def use_input_param(self, name: str = "input") -> Self:
        """
        Sets the name of the input parameter.

        :param name: The name to set for the input parameter. Defaults to 'input'.
        :return: The ChainFactory instance for method chaining.
        """
        self.__param_input = name
        self.__logger.debug("Setting input parameter name to '%s'", self.__param_input)
        return self

    def use_format_instructions_param(self, name: str = "format_instructions") -> Self:
        """
        Sets the name of the format instructions parameter.

        :param name: The name to set for the format instructions parameter.
        Defaults to 'format_instructions'.
        :return: The ChainFactory instance for method chaining.
        """
        self.__param_format_instructions = name
        self.__logger.debug(
            "Setting format instructions parameter name to '%s'",
            self.__param_format_instructions,
        )
        return self

    def use_prompt_template(
        self, prompt_template: BasePromptTemplate[PromptOutputVar]
    ) -> Self:
        """
        Sets the prompt template instance.

        :param prompt_template: An instance of PromptTemplate to set.
        :return: The ChainFactory instance for method chaining.
        """
        self.__prompt_template = prompt_template
        return self

    def use_output_transformer(
        self, output_parser: RunnableGenerator[LanguageModelOutputVar, ChainOutputVar]
    ) -> Self:
        """
        Sets the output transformer instance.

        :param output_parser: An instance of RunnableGenerator for output transformation.
        If None, a default StrOutputParser is used.
        :return: The ChainFactory instance for method chaining.
        """
        self.__output_transformer = output_parser
        self.__logger.debug("Setting output transformer: %s", self.__output_transformer)
        return self

    def build_raw_chain(
        self,
    ) -> RunnableSerializable[ChainInputType, ChainOutputVar]:
        chain = (
            self.input_transformer
            | self.prompt_template
            | self.language_model
            | self.output_transformer
        )
        return chain
