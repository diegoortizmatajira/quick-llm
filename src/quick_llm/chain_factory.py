"""Factory class for managing language model instances."""

import logging
from typing import AsyncIterator, Callable, Generic, Iterator, Self, cast, overload

from langchain_core.language_models import (
    LanguageModelLike,
    LanguageModelOutput,
)
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.string import PromptTemplateFormat
from langchain_core.runnables import Runnable, RunnableGenerator, RunnableLambda
from langchain_core.runnables.base import RunnableSerializable
from pydantic import BaseModel

from quick_llm.prompt_input_parser import PromptInputParser
from quick_llm.type_definitions import ChainInputType, ChainOutputVar


# pylint: disable=too-many-instance-attributes disable=too-many-public-methods
class ChainFactory(Generic[ChainOutputVar]):
    """Factory class for managing language model instances."""

    def __init__(
        self,
        output_type: type[ChainOutputVar] = str,
    ) -> None:
        # Logger setup
        self.__logger = logging.getLogger(__name__)
        self.__detailed_logging: bool = False
        # Transformers (Input/Output)
        self.__in_transf: Runnable[ChainInputType, dict] | None = None
        self.__out_transf: Runnable[LanguageModelOutput, ChainOutputVar] | None = None
        # Output cleaner function
        self.__out_cleaner: Callable[[str], str] = self.default_cleaner_function
        # LLM components
        self.__language_model: LanguageModelLike | None = None
        self.__prompt_template: BasePromptTemplate[PromptValue] | None = None
        # Parameter names
        self.__param_input: str = "input"
        self.__param_format_instructions: str = "format_instructions"
        # JSON model for output parsing
        self.__json_model: type[BaseModel] | None = None
        # Conditional initialization based on output_type
        self.__output_type = output_type

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

    def default_cleaner_function(self, text: str) -> str:
        """
        Default function to clean the output text.

        :param text: The text to be cleaned.
        :return: The cleaned text.
        """
        return text.replace("\\_", "_")

    @staticmethod
    def get_readable_value(value: object) -> object:
        """
        Converts the input object into a human-readable format.

        :param value: The object to be converted. This can be a BaseMessage, BaseModel, or other types.
        :return: A human-readable representation of the object.
        """
        # WARN: If there are non-serializable objects, this method should be updated to handle them or it will fail
        if isinstance(value, BaseMessage):
            return value.model_dump_json(indent=2)
        if isinstance(value, BaseModel):
            return value.model_dump_json(indent=2)
        # elif isinstance(value, dict):
        #     return json.dumps(value, indent=2)
        return value

    def passthrough_logger[T](self, caption: str) -> Runnable[T, T]:
        """Captures the outputs and logs it. It is included in the default implementation of `wrap_chain` method"""

        def output_collector(output: Iterator[T]) -> Iterator[T]:
            for item in output:
                self.__logger.debug(f"{caption}: %s", self.get_readable_value(item))
                yield item

        async def aoutput_collector(output: AsyncIterator[T]) -> AsyncIterator[T]:
            async for item in output:
                self.__logger.debug(f"{caption}: %s", self.get_readable_value(item))
                yield item

        return RunnableGenerator(output_collector, aoutput_collector)

    def wrap[Input, Output](
        self, runnable: Runnable[Input, Output], caption: str
    ) -> Runnable[Input, Output]:
        """
        Wraps a runnable with detailed logging if enabled.

        :param runnable: The runnable to be wrapped.
        :return: The wrapped runnable with logging if detailed logging is enabled.
        """
        if self.__detailed_logging:
            return runnable | self.passthrough_logger(caption)
        return runnable

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
    def input_transformer(self) -> Runnable[ChainInputType, dict]:
        """
        Gets the input transformer instance.

        :return: The current instance of Runnable for input transformation.
        """
        if self.__in_transf is None:
            self.__in_transf = PromptInputParser(self.__param_input)
        return self.__in_transf

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
    def output_cleaner(
        self,
    ) -> Runnable[LanguageModelOutput, LanguageModelOutput]:
        """
        This function is used to clean the output messages from invalid escape sequences.
        It is included in the default implementation of chains to ensure the output is valid.
        """

        def clean_item(item: LanguageModelOutput) -> LanguageModelOutput:
            if isinstance(item, BaseMessage):
                if isinstance(item.content, str):
                    item.content = self.__out_cleaner(item.content)
                elif isinstance(item.content, list):
                    item.content = [
                        self.__out_cleaner(item) if isinstance(item, str) else item
                        for item in item.content
                    ]
            if isinstance(item, str):
                item = self.__out_cleaner(item)
            return item

        def clean_generator(
            output_values: Iterator[LanguageModelOutput],
        ) -> Iterator[LanguageModelOutput]:
            for item in output_values:
                yield clean_item(item)

        async def aclean_generator(
            output_values: AsyncIterator[LanguageModelOutput],
        ) -> AsyncIterator[LanguageModelOutput]:
            async for item in output_values:
                yield clean_item(item)

        return RunnableGenerator(clean_generator, aclean_generator)

    @property
    def output_transformer(
        self,
    ) -> Runnable[LanguageModelOutput, ChainOutputVar]:
        """
        Gets the output transformer instance.

        :return: The current instance of Runnable for output transformation.
        """
        if self.__out_transf:
            return self.__out_transf
        if self.__output_type is str:
            self.use_output_transformer(
                cast(
                    Runnable[LanguageModelOutput, ChainOutputVar],
                    StrOutputParser(),
                )
            )
            # Calls recursively to return the newly set transformer
            return self.output_transformer

        raise self.__fail("Output transformer is not set.")

    def use(self, visitor: Callable[[Self], None]) -> Self:
        """
        Applies a visitor function to the ChainFactory instance.

        :param visitor: A callable that takes a ChainFactory instance and returns None.
        :return: The ChainFactory instance for method chaining.
        """
        self.__logger.debug("Applying visitor to ChainFactory")
        visitor(self)
        return self

    def use_detailed_logging(self, enable: bool = True) -> Self:
        """
        Enables or disables detailed logging for the ChainFactory.

        :param enable: A boolean flag to enable or disable detailed logging. Defaults to True.
        :return: The ChainFactory instance for method chaining.
        """
        self.__detailed_logging = enable
        self.__logger.debug("Setting detailed logging to %s", self.__detailed_logging)
        return self

    def use_language_model(self, language_model: LanguageModelLike) -> Self:
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
        self, output_parser: Runnable[LanguageModelOutput, ChainOutputVar]
    ) -> Self:
        """
        Sets the output transformer instance.

        :param output_parser: An instance of Runnable for output transformation.
        If None, a default StrOutputParser is used.
        :return: The ChainFactory instance for method chaining.
        """
        self.__out_transf = output_parser
        self.__logger.debug("Setting output transformer: %s", self.__out_transf)
        return self

    def use_custom_output_cleaner(self, cleaner_function: Callable[[str], str]) -> Self:
        """
        Sets a custom output cleaner function.

        :param cleaner_function: A callable that takes a string and returns a cleaned string.
        :return: The ChainFactory instance for method chaining.
        """
        self.__out_cleaner = cleaner_function
        self.__logger.debug("Setting custom output cleaner function.")
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
            self.wrap(self.input_transformer, "Input Transformer")
            | self.wrap(self.additional_values_injector, "Additional Values Injector")
            | self.wrap(self.prompt_template, "Prompt Template")
            | self.wrap(self.language_model, "Language Model")
            | self.wrap(self.output_cleaner, "Output Cleaner")
            | self.wrap(self.output_transformer, "Output Transformer")
        )
        return chain
