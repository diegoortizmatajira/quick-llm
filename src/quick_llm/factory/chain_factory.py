"""Factory class for managing language model instances."""

from typing import (
    Any,
    AsyncIterator,
    Callable,
    Iterator,
    Self,
    Type,
    cast,
    get_origin,
    overload,
    override,
)

from langchain_core.documents import Document
from langchain_core.language_models import (
    LanguageModelOutput,
)
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import (
    Runnable,
    RunnableAssign,
    RunnableGenerator,
    RunnableLambda,
)

from ..support import (
    BaseFactory,
    PromptInputParser,
    ChainInputType,
    ChainOutputVar,
    ModelTypeVar,
    Strategy,
)
from ..strategies import (
    TextStrategy,
    DictModelStrategy,
    TypedModelStrategy,
    NullStrategy,
)


# pylint: disable=too-many-instance-attributes disable=too-many-public-methods
class ChainFactory(BaseFactory[ChainOutputVar, ModelTypeVar]):
    """Factory class for managing language model instances."""

    def __init__(
        self,
        output_type: type[ChainOutputVar] = str,
        structured_model_type: type[ModelTypeVar] | None = None,
    ) -> None:
        super().__init__(
            output_type=output_type,
            structured_model_type=structured_model_type,
        )
        # Transformers (Input/Output)
        self._in_transf: Runnable[ChainInputType, dict] | None = None
        self._out_transf: Runnable[LanguageModelOutput, ChainOutputVar] | None = None
        # Customizable behaviors
        self._strategies: list[Type[Strategy[Any]]] = [
            TextStrategy,
            DictModelStrategy,
            TypedModelStrategy,
            NullStrategy,
        ]
        self.__strategy: Strategy | None = None
        self.__out_cleaner: Callable[[str], str] = self.default_cleaner_function
        self.__ctx_formatter: Callable[[list[Document]], str] = (
            self.default_context_formatter
        )
        self.__retrieval_query_builder: Callable[[dict], str] = lambda x: x.get(
            self.__param_input, ""
        )
        self.__doc_refs_formatter: Callable[[list[Document]], str] = (
            self.default_references_formatter
        )
        # Parameter names
        self.__param_input: str = "input"
        self.__param_format_instructions: str = "format_instructions"
        self.__param_context: str = "context"
        # Custom response_keys
        self.__answer_key: str = "answer"
        self.__source_documents_key: str = "source_documents"
        # JSON model for output parsing
        self.__structured_output_model: type[ModelTypeVar] | None = None
        # usage flags
        self.__rag_return_sources: bool = False
        self.__rag_return_sources_formatted_as_string: bool = False
        self._logger.debug("Initialized ChainFactory with output type: %s", output_type)

    @staticmethod
    def for_structured_output(
        structured_model_type: type[ModelTypeVar],
    ) -> "ChainFactory[dict[str, object], ModelTypeVar]":
        """
        Creates a ChainFactory instance based on a given JSON model.

        :param json_model: A Pydantic BaseModel class that will be used to interpret JSON outputs.
        :return: A ChainFactory instance configured to use the provided JSON model.
        """
        return ChainFactory(dict[str, object], structured_model_type)

    @staticmethod
    def for_typed_output(
        structured_model_type: type[ModelTypeVar],
    ) -> "ChainFactory[ModelTypeVar, ModelTypeVar]":
        """
        Creates a ChainFactory instance based on a given JSON model.

        :param json_model: A Pydantic BaseModel class that will be used to interpret JSON outputs.
        :return: A ChainFactory instance configured to use the provided JSON model.
        """
        return ChainFactory(structured_model_type, structured_model_type)

    @staticmethod
    def for_rag_with_sources() -> "ChainFactory[dict[str, object],None]":
        """
        Creates a ChainFactory instance based on a given JSON model.

        :param json_model: A Pydantic BaseModel class that will be used to interpret JSON outputs.
        :return: A ChainFactory instance configured to use the provided JSON model.
        """
        result = ChainFactory(dict[str, object]).use_rag_returning_sources(True)
        # if json_model:
        #     result.use_structured_output(json_model)
        return result

    def default_cleaner_function(self, text: str) -> str:
        """
        Default function to clean the output text.

        :param text: The text to be cleaned.
        :return: The cleaned text.
        """
        return text.replace("\\_", "_")

    def default_context_formatter(self, documents: list[Document]) -> str:
        """
        Default function to format context from a list of documents.

        :param documents: A list of Document instances.
        :return: A formatted string representing the context.
        """
        return "\n\n".join(doc.page_content for doc in documents)

    def default_references_formatter(self, documents: list[Document]) -> str:
        """
        Default function to format references from a list of documents.

        :param documents: A list of Document instances.
        :return: A formatted string representing the references.
        """
        return "\n\nReferences:\n\n" + "\n\n".join(
            [
                f"**[{i + 1}]** {source.metadata.get('source', None) or source.page_content}"
                for i, source in enumerate(documents)
            ]
        )

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
    def structured_output_model(self) -> type[ModelTypeVar] | None:
        """
        Gets the structured output model.

        :return: The current Pydantic BaseModel class for structured output or None if not set.
        """
        return self.__structured_output_model

    @property
    def strategy(self) -> Strategy:
        """
        Gets the strategy instance.

        :return: The current instance of Strategy or None if not set.
        """
        if self.__strategy is None:
            raise self._fail("Strategy is not set.")
        return self.__strategy

    @property
    def input_transformer(self) -> Runnable[ChainInputType, dict]:
        """
        Gets the input transformer instance.

        :return: The current instance of Runnable for input transformation.
        """
        if self._in_transf is None:
            self._in_transf = PromptInputParser(self.__param_input)
        return self._in_transf

    @property
    def uses_rag(self) -> bool:
        """
        Indicates whether Retrieval-Augmented Generation (RAG) is enabled.

        :return: True if RAG is enabled, False otherwise.
        """
        return self._retriever is not None

    @property
    def additional_values_injector(self) -> Runnable[dict, dict]:
        """
        Provides a lambda function that injects additional values into the
        existing input dictionary.

        This method creates a dictionary of additional values to be passed into
        the chain. If the JSON model is being used and the output transformer
        is of the type JsonOutputParser, it adds format instructions specific
        to the JSON model to the `additional_values` dictionary. The lambda
        function merges the existing input dictionary with these additional
        values.

        :return: A Runnable instance that injects additional values into the input dictionary.
        """
        additional_values: dict[str, object] = {}

        output_transformer = self.output_transformer

        if self.__structured_output_model and isinstance(
            output_transformer, JsonOutputParser
        ):
            # Adds format instructions for JSON model if applicable
            self._logger.debug(
                "Building chain with JSON model: %s", self.__structured_output_model
            )
            additional_values[self.format_instructions_param] = (
                output_transformer.get_format_instructions()
            )
            self._logger.debug(
                "Added format instructions to chain: %s", additional_values
            )

        # Returns an injector for additional values
        return RunnableLambda[dict, dict](
            lambda x: {**x, **additional_values}, name="Additional Values Injector"
        )

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

        return RunnableGenerator(
            clean_generator, aclean_generator, name="Default Output Cleaner"
        )

    @property
    def output_transformer(
        self,
    ) -> Runnable[LanguageModelOutput, ChainOutputVar]:
        """
        Gets the output transformer instance.

        :return: The current instance of Runnable for output transformation.
        """
        if self._out_transf:
            return self._out_transf
        if self.__structured_output_model is None:
            self.use_output_transformer(
                cast(
                    Runnable[LanguageModelOutput, ChainOutputVar],
                    StrOutputParser(name="Text Output Parser"),
                )
            )
            # Calls recursively to return the newly set transformer
            return self.output_transformer

        raise self._fail("Output transformer is not set.")

    @property
    def document_formatter(self) -> Runnable[list[Document], str]:
        """
        Allows the context retrieval to be formatted as a string to be passed down to the prompt.
        """

        def format_docs(docs: list[Document]) -> str:
            for i, doc in enumerate(docs):
                self._logger.debug("Recovered document (%d): %s", i, doc)
            return self.__ctx_formatter(docs)

        def formatter_function(input_docs: Iterator[list[Document]]) -> Iterator[str]:
            for docs in input_docs:
                yield format_docs(docs)

        async def aformatter_function(
            input_docs: AsyncIterator[list[Document]],
        ) -> AsyncIterator[str]:
            async for docs in input_docs:
                yield format_docs(docs)

        return RunnableGenerator(
            formatter_function, aformatter_function, name="Document Formatter"
        )

    @property
    def final_answer_formatter(self) -> Runnable[dict, str]:
        """
        Returns the final answer formatted along with the source references in
        a single string.
        """

        def formatter(answers: Iterator[dict]) -> Iterator[str]:
            references_text: str | None = None
            for answer in answers:
                # If the answer contains the answer key, then it streams its content
                if answer.get(self.__answer_key, None):
                    yield answer[self.__answer_key]
                # If the answer contains the documents key, keep it until it
                # finishes streaming the answer
                if answer.get(self.__source_documents_key, None):
                    docs = cast(list[Document], answer[self.__source_documents_key])
                    references_text = self.__doc_refs_formatter(docs)
            # If it has a generated references_text, then send it to the output
            if references_text:
                yield references_text

        async def aformatter(answers: AsyncIterator[dict]) -> AsyncIterator[str]:
            references_text: str | None = None
            async for answer in answers:
                # If the answer contains the answer key, then it streams its content
                if answer.get(self.__answer_key, None):
                    yield answer[self.__answer_key]
                # If the answer contains the documents key, keep it until it
                # finishes streaming the answer
                if answer.get(self.__source_documents_key, None):
                    docs = cast(list[Document], answer[self.__source_documents_key])
                    references_text = self.__doc_refs_formatter(docs)
            # If it has a generated references_text, then send it to the output
            if references_text:
                yield references_text

        return RunnableGenerator(formatter, aformatter, name="Final Answer Formatter")

    @property
    def answer_key(self) -> str:
        """
        Gets the name of the answer key in the output.

        :return: The name of the answer key.
        """
        return self.__answer_key

    @property
    def document_references_key(self) -> str:
        """
        Gets the name of the document references key in the output.

        :return: The name of the document references key.
        """
        return self.__source_documents_key

    def use_input_param(self, name: str = "input") -> Self:
        """
        Sets the name of the input parameter.

        :param name: The name to set for the input parameter. Defaults to 'input'.
        :return: The ChainFactory instance for method chaining.
        """
        self.__param_input = name
        self._logger.debug("Setting input parameter name to '%s'", self.__param_input)
        return self

    def use_format_instructions_param(self, name: str = "format_instructions") -> Self:
        """
        Sets the name of the format instructions parameter.

        :param name: The name to set for the format instructions parameter.
        Defaults to 'format_instructions'.
        :return: The ChainFactory instance for method chaining.
        """
        self.__param_format_instructions = name
        self._logger.debug(
            "Setting format instructions parameter name to '%s'",
            self.__param_format_instructions,
        )
        return self

    def use_context_param(self, name: str = "context") -> Self:
        """
        Sets the name of the context parameter.

        :param name: The name to set for the context parameter. Defaults to 'context'.
        :return: The ChainFactory instance for method chaining.
        """
        self.__param_context = name
        self._logger.debug(
            "Setting context parameter name to '%s'", self.__param_context
        )
        return self

    def use_answer_key(self, name: str = "answer") -> Self:
        """
        Sets the name of the answer key in the output.

        :param name: The name to set for the answer key. Defaults to 'answer'.
        :return: The ChainFactory instance for method chaining.
        """
        self.__answer_key = name
        self._logger.debug("Setting answer key name to '%s'", self.__answer_key)
        return self

    def use_structured_output(self, model: type[ModelTypeVar]) -> Self:
        """
        Sets the JSON model for output parsing.

        :param model: A Pydantic BaseModel class to parse the output into.
        :return: The ChainFactory instance for method chaining.
        """
        self.__structured_output_model = model
        self.use_output_transformer(
            cast(
                Runnable[LanguageModelOutput, ChainOutputVar],
                JsonOutputParser(
                    pydantic_object=self.__structured_output_model,
                    name=f"Json outputParser for {self.structured_output_model}",
                ),
            )
        )
        self._logger.debug(
            "Setting JSON model for output parsing: %s", self.__structured_output_model
        )
        return self

    @overload
    def use_output_transformer(
        self, output_parser: Runnable[LanguageModelOutput, ChainOutputVar]
    ) -> Self:
        """
        Sets the output transformer instance.

        :param output_parser: An instance of Runnable for output transformation.
        If None, a default StrOutputParser is used.
        :return: The ChainFactory instance for method chaining.
        """

    @overload
    def use_output_transformer(
        self, output_parser: Callable[[LanguageModelOutput], ChainOutputVar]
    ) -> Self:
        """
        Sets the output transformer instance.

        :param output_parser: An instance of Callable for output transformation.
        If None, a default StrOutputParser is used.
        :return: The ChainFactory instance for method chaining.
        """

    def use_output_transformer(
        self,
        output_parser: Runnable[LanguageModelOutput, ChainOutputVar]
        | Callable[[LanguageModelOutput], ChainOutputVar],
    ) -> Self:
        """
        Sets the output transformer instance.

        :param output_parser: An instance of Runnable for output transformation.
        If None, a default StrOutputParser is used.
        :return: The ChainFactory instance for method chaining.
        """
        if isinstance(output_parser, Callable):
            output_parser = RunnableLambda(output_parser, name="Custom Output Parser")
        self._out_transf = output_parser
        self._logger.debug("Setting output transformer: %s", self._out_transf)
        return self

    def use_custom_output_cleaner(self, cleaner_function: Callable[[str], str]) -> Self:
        """
        Sets a custom output cleaner function.

        :param cleaner_function: A callable that takes a string and returns a cleaned string.
        :return: The ChainFactory instance for method chaining.
        """
        self.__out_cleaner = cleaner_function
        self._logger.debug("Setting custom output cleaner function.")
        return self

    def use_custom_context_formatter(
        self, formatter_function: Callable[[list[Document]], str]
    ) -> Self:
        """
        Sets a custom context formatter function.

        :param formatter_function: A callable that takes a list of Document instances
        and returns a formatted string.
        :return: The ChainFactory instance for method chaining.
        """
        self.__ctx_formatter = formatter_function
        self._logger.debug("Setting custom context formatter function.")
        return self

    def use_custom_retrieval_query_builder(
        self, query_builder_function: Callable[[dict], str]
    ) -> Self:
        """
        Sets a custom retrieval query builder function.

        :param query_builder_function: A callable that takes a dictionary of input values
        and returns a query string.
        :return: The ChainFactory instance for method chaining.
        """
        self.__retrieval_query_builder = query_builder_function
        self._logger.debug("Setting custom retrieval query builder function.")
        return self

    def use_rag_returning_sources(
        self, returning_sources: bool, format_as_string: bool = False
    ) -> Self:
        """
        Sets whether the RAG component should return source documents along
        with the generated answer.

        :param returning_sources: A boolean flag to indicate if sources should be returned.
        :return: The ChainFactory instance for method chaining.
        """
        self.__rag_return_sources = returning_sources
        self.__rag_return_sources_formatted_as_string = format_as_string
        self._logger.debug(
            "Setting RAG returning sources to %s", self.__rag_return_sources
        )
        return self

    def select_strategy(self):
        """
        Selects and initializes the appropriate strategy based on the output type
        and structured model type.
        """
        for strategy_cls in self._strategies:
            strategy = strategy_cls.should_be_selected(self)
            if strategy:
                self._logger.debug(
                    "Selecting %s for output type '%s' and structured model type '%s'.",
                    strategy_cls.__name__,
                    self._output_type,
                    self.structured_model_type,
                )
                self.__strategy = strategy
                return

    def __build_without_rag(
        self,
    ) -> Runnable[ChainInputType, ChainOutputVar]:
        """
        Constructs and returns the complete runnable chain without RAG components.

        The chain consists of the following components, connected sequentially:
        - Input transformer: Transforms raw input into a structured format.
        - Additional values injector: Injects additional parameters required for the chain.
        - Prompt template: Generates the prompt based on the transformed input.
        - Language model: Generates an output based on the prompt.
        - Output transformer: Parses and transforms the model output into the desired format.

        :return: A Runnable instance representing the complete chain.
        """
        # if self.__structured_output_model is None:
        #     self.__strategy = TextStrategy(cast(ChainFactory[Any, None], self))
        # else:
        #     self.__strategy = DictModelStrategy(
        #         cast(ChainFactory[Any, ModelTypeVar], self)
        #     )
        # adapted_language_model = self.strategy.adapted_llm

        chain = (
            self.wrap(self.input_transformer, "Input Transformer")
            | self.wrap(self.additional_values_injector, "Additional Values Injector")
            | self.wrap(self.prompt_template, "Prompt Template")
            | self.wrap(self.strategy.adapted_llm, "Strategy Adapted LLM")
            # | self.wrap(self.language_model, "Language Model")
            # | self.wrap(self.output_cleaner, "Output Cleaner")
            # | self.wrap(self.output_transformer, "Output Transformer")
        )
        self._logger.debug(
            "Built chain without RAG components: %s", self._pretty_runnable(chain)
        )
        return chain

    def __build_with_rag(self) -> Runnable[ChainInputType, ChainOutputVar]:
        chain = (
            self.wrap(self.input_transformer, "Input Transformer")
            | RunnableAssign(
                {
                    # Selects the value to use to retrieve documents from the store
                    self.__param_context: self.__retrieval_query_builder
                    # Retrieves the documents
                    | self.wrap(self.retriever, "Retriever")
                    # Formats the documents into a single string
                    | self.wrap(self.document_formatter, "Document Formatter")
                },  # type: ignore
                name="Retrieve and Format Documents",
            )
            | self.wrap(self.additional_values_injector, "Additional Values Injector")
            | self.wrap(self.prompt_template, "Prompt Template")
            | self.wrap(self.language_model, "Language Model")
            | self.wrap(self.output_cleaner, "Output Cleaner")
            | self.wrap(self.output_transformer, "Output Transformer")
        )
        self._logger.debug(
            "Built chain with RAG components: %s", self._pretty_runnable(chain)
        )
        return chain

    def __build_with_rag_with_sources(
        self,
    ) -> Runnable[ChainInputType, ChainOutputVar]:
        if (
            self.__rag_return_sources_formatted_as_string
            and self.__structured_output_model is not None
        ):
            raise self._fail(
                "Cannot combine returning sources formatted as string with JSON model output."
            )
        chain = (
            self.wrap(self.input_transformer, "Input Transformer")
            # Retrieves the documents and keep them in the source_documents_key
            | RunnableAssign(
                {
                    # Selects the value to use to retrieve documents from the store
                    self.__source_documents_key: self.__retrieval_query_builder
                    # Retrieves the documents
                    | self.wrap(self.retriever, "Retriever")
                },  # type: ignore
                name="Retrieve Documents",
            )
            # Builds the answer value by executing the RAG
            | RunnableAssign(
                {
                    self.__answer_key: (
                        # Builds the context variable content
                        RunnableAssign(
                            {
                                self.__param_context: (
                                    # Selects the value to use to retrieve documents from the store
                                    (lambda x: x.get(self.__source_documents_key, []))
                                    # Formats the documents into a single string
                                    | self.wrap(
                                        self.document_formatter, "Document Formatter"
                                    )
                                ),
                            },  # type: ignore
                            name="Build Context from Retrieved Documents",
                        )
                        | self.wrap(
                            self.additional_values_injector,
                            "Additional Values Injector",
                        )
                        | self.wrap(self.prompt_template, "Prompt Template")
                        | self.wrap(self.language_model, "Language Model")
                        | self.wrap(self.output_cleaner, "Output Cleaner")
                        | self.wrap(self.output_transformer, "Output Transformer")
                    )
                },
                name="Generate Answer along with Sources",
            )
        )
        if self.__rag_return_sources_formatted_as_string:
            # Formats the source documents as a single string
            chain = chain | self.wrap(
                self.final_answer_formatter, "Final Answer Formatter"
            )
        self._logger.debug(
            "Built chain with RAG components and document references: %s",
            self._pretty_runnable(chain),
        )
        # INFO: uses a cast to avoid LSP error about incompatible types
        return cast(Runnable[ChainInputType, ChainOutputVar], chain)

    @override
    def build(
        self,
    ) -> Runnable[ChainInputType, ChainOutputVar]:
        """
        Constructs and returns the complete runnable chain, either with or without
        Retrieval-Augmented Generation (RAG) components based on the current configuration.

        If RAG is enabled (`use_rag`), the chain handles retrieval and integration
        of external context documents into the generation process. If `rag_return_sources`
        is set, it ensures source documents are included in the output.

        :return: A RunnableSerializable instance representing the complete chain.
        """
        self._logger.info("Building chain")
        self.select_strategy()
        chain = (
            self.wrap(self.input_transformer, "Input Transformer")
            | self.wrap(self.additional_values_injector, "Additional Values Injector")
            | self.wrap(self.prompt_template, "Prompt Template")
            | self.wrap(self.strategy.adapted_llm, "Strategy Adapted LLM")
            # | self.wrap(self.language_model, "Language Model")
            # | self.wrap(self.output_cleaner, "Output Cleaner")
            # | self.wrap(self.output_transformer, "Output Transformer")
        )
        self._logger.debug("Built chain: %s", self._pretty_runnable(chain))
        return chain
        # if self.uses_rag:
        #     if self.__rag_return_sources:
        #         return self.__build_with_rag_with_sources()
        #     return self.__build_with_rag()
        # return self.__build_without_rag()
