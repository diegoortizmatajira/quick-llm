"""Factory class for managing language model instances."""

import logging
from typing import Callable, Generic, Self, cast, overload

from langchain_core.language_models import (
    BaseLanguageModel,
    LanguageModelLike,
    LanguageModelOutput,
)
from langchain_core.language_models.base import LanguageModelOutputVar
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.string import PromptTemplateFormat
from langchain_core.runnables import RunnableGenerator, RunnableLambda
from langchain_core.runnables.base import RunnableSerializable
from pydantic import BaseModel

from quick_llm.prompt_input_parser import PromptInputParser
from quick_llm.type_definitions import ChainInputType, ChainOutputVar


class ChainFactory(Generic[ChainOutputVar]):
    """Factory class for managing language model instances."""

    def __init__(
        self,
        output_type: type[ChainOutputVar] = str,
    ) -> None:
        self.__logger = logging.getLogger(__name__)
        self.__input_transformer: RunnableGenerator[ChainInputType, dict] | None = None
        self.__output_transformer: (
            RunnableSerializable[LanguageModelOutput, ChainOutputVar] | None
        ) = None
        self.__language_model: LanguageModelLike | None = None
        self.__prompt_template: BasePromptTemplate[PromptValue] | None = None
        self.__param_input: str = "input"
        self.__param_format_instructions: str = "format_instructions"
        self.__json_model: type[BaseModel] | None = None
        # Conditional initialization based on output_type
        self.__output_type = output_type
        if self.__output_type is str:
            self.use_output_transformer(
                cast(
                    RunnableSerializable[LanguageModelOutput, ChainOutputVar],
                    StrOutputParser(),
                )
            )

    @staticmethod
    def for_json_model(
        json_model: type[BaseModel],
    ) -> "ChainFactory[dict[str, object]]":
        """
        Creates a ChainFactory instance based on a given JSON model.

        :param json_model: A Pydantic BaseModel class that will be used to interpret JSON outputs.
        :return: A ChainFactory instance configured to use the provided JSON model.
        """
        return ChainFactory(dict[str, object]).use_json_model(json_model)

    def __fail(self, message: str) -> Exception:
        self.__logger.error(message)
        return RuntimeError(message)

    @property
    def language_model(self) -> LanguageModelLike:
        """
        Gets the language model instance.

        :return: The current instance of BaseLanguageModel or None if not set.
        """
        if self.__language_model is None:
            raise self.__fail("Language model is not set.")
        return self.__language_model

    @property
    def prompt_template(self) -> BasePromptTemplate[PromptValue]:
        """
        Gets the prompt template instance.

        :return: The current instance of PromptTemplate or None if not set.
        """
        if self.__prompt_template is None:
            raise self.__fail("Prompt template is not set.")
        return self.__prompt_template

    @property
    def input_param(self) -> str:
        """
        Gets the name of the input parameter.

        :return: The name of the input parameter.
        """
        return self.__param_input

    @property
    def format_instructions_param(self) -> str:
        """
        Gets the name of the format instructions parameter.

        :return: The name of the format instructions parameter.
        """
        return self.__param_format_instructions

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
    def additional_values_injector(self) -> RunnableLambda[dict, dict]:
        """
        Provides a lambda function that injects additional values into the existing input dictionary.

        This method creates a dictionary of additional values to be passed into the chain. If the JSON model
        is being used and the output transformer is of the type JsonOutputParser, it adds format instructions
        specific to the JSON model to the `additional_values` dictionary. The lambda function merges the
        existing input dictionary with these additional values.

        :return: A RunnableLambda instance that injects additional values into the input dictionary.
        """
        additional_values: dict[str, object] = {}

        output_transformer = self.output_transformer

        if self.__json_model and isinstance(output_transformer, JsonOutputParser):
            # Adds format instructions for JSON model if applicable
            self.__logger.debug("Building chain with JSON model: %s", self.__json_model)
            additional_values[self.format_instructions_param] = (
                output_transformer.get_format_instructions()
            )
            self.__logger.debug(
                "Added format instructions to chain: %s", additional_values
            )

        # Returns an injector for additional values
        return RunnableLambda[dict, dict](lambda x: {**x, **additional_values})

    @property
    def output_transformer(
        self,
    ) -> RunnableSerializable[LanguageModelOutputVar, ChainOutputVar]:
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

    @overload
    def use_prompt_template(
        self, prompt_template: BasePromptTemplate[PromptValue]
    ) -> Self:
        """
        Sets the prompt template instance.

        :param prompt_template: An instance of PromptTemplate to set.
        :return: The ChainFactory instance for method chaining.
        """

    @overload
    def use_prompt_template(
        self,
        prompt_template: str,
        prompt_template_format: PromptTemplateFormat = "f-string",
        partial_variables: dict[str, str] | None = None,
    ) -> Self:
        """
        Sets the prompt template instance from a string.

        :param prompt_template: A string representing the prompt template.
        :param prompt_template_format: The format of the prompt template string.
        :param partial_variables: A dictionary of partial variables for the prompt template.
        :return: The ChainFactory instance for method chaining.
        """

    def use_prompt_template(
        self,
        prompt_template: str | BasePromptTemplate[PromptValue],
        prompt_template_format: PromptTemplateFormat = "f-string",
        partial_variables: dict[str, str] | None = None,
    ) -> Self:
        """
        Sets the prompt template instance.
        :param prompt_template: An instance of PromptTemplate or a string representing
        the prompt template.
        :param prompt_template_format: The format of the prompt template string.
        :param partial_variables: A dictionary of partial variables for the prompt template.
        :return: The ChainFactory instance for method chaining.
        """
        if isinstance(prompt_template, str):
            # Creates a PromptTemplate from string
            prompt_template = PromptTemplate.from_template(
                template=prompt_template,
                template_format=prompt_template_format,
                partial_variables=partial_variables,
            )
        self.__prompt_template = prompt_template
        return self

    def use_json_model(self, model: type[BaseModel]) -> Self:
        """
        Sets the JSON model for output parsing.

        :param model: A Pydantic BaseModel class to parse the output into.
        :return: The ChainFactory instance for method chaining.
        """
        self.__json_model = model
        self.use_output_transformer(
            cast(
                RunnableSerializable[LanguageModelOutput, ChainOutputVar],
                JsonOutputParser(pydantic_object=self.__json_model),
            )
        )
        self.__logger.debug(
            "Setting JSON model for output parsing: %s", self.__json_model
        )
        return self

    def use_output_transformer(
        self, output_parser: RunnableSerializable[LanguageModelOutput, ChainOutputVar]
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

    def build(
        self,
    ) -> RunnableSerializable[ChainInputType, ChainOutputVar]:
        """
        Constructs and returns the complete runnable chain.

        The chain consists of the following components, connected sequentially:
        - Input transformer: Transforms raw input into a structured format.
        - Additional values injector: Injects additional parameters required for the chain.
        - Prompt template: Generates the prompt based on the transformed input.
        - Language model: Generates an output based on the prompt.
        - Output transformer: Parses and transforms the model output into the desired format.

        :return: A RunnableSerializable instance representing the complete chain.
        """
        chain = (
            self.input_transformer
            | self.additional_values_injector
            | self.prompt_template
            | self.language_model
            | self.output_transformer
        )
        return chain
