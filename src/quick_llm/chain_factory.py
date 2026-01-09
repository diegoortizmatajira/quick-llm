"""Factory class for managing language model instances."""

import logging
from typing import (
    AsyncIterator,
    Callable,
    Generic,
    Iterator,
    Self,
    cast,
    overload,
)

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import (
    LanguageModelLike,
    LanguageModelOutput,
)
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.prompts.string import PromptTemplateFormat
from langchain_core.retrievers import RetrieverLike
from langchain_core.runnables import (
    Runnable,
    RunnableGenerator,
    RunnableLambda,
    RunnablePassthrough,
    RunnableSerializable,
)
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TextSplitter,
    TokenTextSplitter,
)
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
        # Customizable behaviors
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
        # LLM components
        self.__language_model: LanguageModelLike | None = None
        self.__prompt_template: BasePromptTemplate[PromptValue] | None = None
        # RAG components
        self.__text_splitter: TextSplitter | None = None
        self.__embeddings: Embeddings | None = None
        self.__vector_store: VectorStore | None = None
        self.__retriever: RetrieverLike | None = None
        # Parameter names
        self.__param_input: str = "input"
        self.__param_format_instructions: str = "format_instructions"
        self.__param_context: str = "context"
        # Custom response_keys
        self.__answer_key: str = "answer"
        self.__source_documents_key: str = "source_documents"
        # JSON model for output parsing
        self.__json_model: type[BaseModel] | None = None
        # usage flags
        self.__use_rag: bool = False
        self.__rag_return_sources: bool = False
        self.__rag_return_sources_formatted_as_string: bool = False
        self.__logger.debug(
            "Initialized ChainFactory with output type: %s", output_type
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

    @staticmethod
    def for_rag_with_sources(
        json_model: type[BaseModel] | None = None,
    ) -> "ChainFactory[dict[str, object]]":
        """
        Creates a ChainFactory instance based on a given JSON model.

        :param json_model: A Pydantic BaseModel class that will be used to interpret JSON outputs.
        :return: A ChainFactory instance configured to use the provided JSON model.
        """
        result = (
            ChainFactory(dict[str, object])
            .use_rag(True)
            .use_rag_returning_sources(True)
        )
        if json_model:
            result.use_json_model(json_model)
        return result

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
        if self.__json_model is None:
            self.use_output_transformer(
                cast(
                    Runnable[LanguageModelOutput, ChainOutputVar],
                    StrOutputParser(),
                )
            )
            # Calls recursively to return the newly set transformer
            return self.output_transformer

        raise self.__fail("Output transformer is not set.")

    @property
    def text_splitter(self) -> TextSplitter:
        """
        Gets the text splitter instance.

        :return: The current instance of TextSplitter.
        """
        if self.__text_splitter is None:
            raise self.__fail("Text splitter is not set.")
        return self.__text_splitter

    @property
    def embeddings(self) -> Embeddings:
        """
        Gets the embeddings instance.

        :return: The current instance of Embeddings.
        """
        if self.__embeddings is None:
            raise self.__fail("Embeddings are not set.")
        return self.__embeddings

    @property
    def vector_store(self) -> VectorStore:
        """
        Gets the vector store instance.

        :return: The current instance of VectorStore.
        """
        if self.__vector_store is None:
            raise self.__fail("Vector store is not set.")
        return self.__vector_store

    @property
    def retriever(self) -> RetrieverLike:
        """
        Gets the retriever instance.

        :return: The current instance of RetrieverLike.
        """
        if self.__retriever is None:
            raise self.__fail("Retriever is not set.")
        return self.__retriever

    @property
    def document_formatter(self) -> Runnable[list[Document], str]:
        """
        Allows the context retrieval to be formatted as a string to be passed down to the prompt.
        """

        def format_docs(docs: list[Document]) -> str:
            for i, doc in enumerate(docs):
                self.__logger.debug("Recovered document (%d): %s", i, doc)
            return self.__ctx_formatter(docs)

        def formatter_function(input_docs: Iterator[list[Document]]) -> Iterator[str]:
            for docs in input_docs:
                yield format_docs(docs)

        async def aformatter_function(
            input_docs: AsyncIterator[list[Document]],
        ) -> AsyncIterator[str]:
            async for docs in input_docs:
                yield format_docs(docs)

        return RunnableGenerator(formatter_function, aformatter_function)

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
                # If the answer contains the documents key, keep it until it finishes streaming the answer
                if answer.get(self.__source_documents_key, None):
                    docs = cast(list[Document], answer[self.__source_documents_key])
                    references_text = self.__doc_refs_formatter(docs)
            # If it has a generated references_text, then send it to the output
            if references_text:
                yield references_text

        return RunnableGenerator(formatter)

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

    def use_context_param(self, name: str = "context") -> Self:
        """
        Sets the name of the context parameter.

        :param name: The name to set for the context parameter. Defaults to 'context'.
        :return: The ChainFactory instance for method chaining.
        """
        self.__param_context = name
        self.__logger.debug(
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
        self.__logger.debug("Setting answer key name to '%s'", self.__answer_key)
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
        self.__logger.debug("Setting custom context formatter function.")
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
        self.__logger.debug("Setting custom retrieval query builder function.")
        return self

    def use_default_token_splitter(
        self, chunk_size: int = 500, chunk_overlap: int = 50
    ) -> Self:
        """
        Sets up a Token TextSplitter with the provided values or the default ones if omitted
        """
        self.__text_splitter = TokenTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return self

    def use_default_text_splitter(
        self, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> Self:
        """
        Sets up a Recursive TextSplitter with the provided values or the default ones if omitted
        """
        self.__text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return self

    def use_rag(self, rag: bool) -> Self:
        """
        Enables or disables the use of Retrieval-Augmented Generation (RAG) in the chain.

        :param rag: A boolean flag to enable or disable RAG.
        :return: The ChainFactory instance for method chaining.
        """
        if self.__use_rag != rag:
            self.__use_rag = rag
            self.__logger.debug("Setting RAG usage to %s", self.__use_rag)
        return self

    def use_rag_returning_sources(
        self, returning_sources: bool, format_as_string: bool = False
    ) -> Self:
        """
        Sets whether the RAG component should return source documents along with the generated answer.

        :param returning_sources: A boolean flag to indicate if sources should be returned.
        :return: The ChainFactory instance for method chaining.
        """
        self.use_rag(True)
        self.__rag_return_sources = returning_sources
        self.__rag_return_sources_formatted_as_string = format_as_string
        self.__logger.debug(
            "Setting RAG returning sources to %s", self.__rag_return_sources
        )
        return self

    def use_embeddings(self, embeddings: Embeddings) -> Self:
        """
        Sets the embeddings instance.

        :param embeddings: An instance of Embeddings to set.
        :return: The ChainFactory instance for method chaining.
        """
        self.__embeddings = embeddings
        self.__logger.debug("Setting embeddings: %s", self.__embeddings)
        return self

    def use_vector_store(self, vector_store: VectorStore) -> Self:
        """
        Sets the vector store instance and enables Retrieval-Augmented Generation (RAG).

        By default, the vector store is also used as a retriever.

        :param vector_store: An instance of VectorStore to set.
        :return: The ChainFactory instance for method chaining.
        """
        self.use_rag(True)
        self.__vector_store = vector_store
        self.__logger.debug("Setting vector store: %s", self.__vector_store)
        # By default, uses the vector store as retriever
        self.__retriever = vector_store.as_retriever()
        return self

    def use_retriever(self, retriever: RetrieverLike) -> Self:
        """
        Sets the retriever instance.

        :param retriever: An instance of RetrieverLike to set.
        :return: The ChainFactory instance for method chaining.
        """
        self.use_rag(True)
        self.__retriever = retriever
        self.__logger.debug("Setting retriever: %s", self.__retriever)
        return self

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

        :return: A RunnableSerializable instance representing the complete chain.
        """
        return (
            self.wrap(self.input_transformer, "Input Transformer")
            | self.wrap(self.additional_values_injector, "Additional Values Injector")
            | self.wrap(self.prompt_template, "Prompt Template")
            | self.wrap(self.language_model, "Language Model")
            | self.wrap(self.output_cleaner, "Output Cleaner")
            | self.wrap(self.output_transformer, "Output Transformer")
        )

    def __build_with_rag(self) -> Runnable[ChainInputType, ChainOutputVar]:
        return (
            self.wrap(self.input_transformer, "Input Transformer")
            | RunnablePassthrough.assign(
                **{
                    # Selects the value to use to retrieve documents from the store
                    self.__param_context: self.__retrieval_query_builder
                    # Retrieves the documents
                    | self.wrap(self.retriever, "Retriever")
                    # Formats the documents into a single string
                    | self.wrap(self.document_formatter, "Document Formatter")
                }
            )
            | self.wrap(self.additional_values_injector, "Additional Values Injector")
            | self.wrap(self.prompt_template, "Prompt Template")
            | self.wrap(self.language_model, "Language Model")
            | self.wrap(self.output_cleaner, "Output Cleaner")
            | self.wrap(self.output_transformer, "Output Transformer")
        )

    def __build_with_rag_with_sources(
        self,
    ) -> Runnable[ChainInputType, ChainOutputVar]:
        if (
            self.__rag_return_sources_formatted_as_string
            and self.__json_model is not None
        ):
            raise self.__fail(
                "Cannot combine returning sources formatted as string with JSON model output."
            )
        chain = (
            self.wrap(self.input_transformer, "Input Transformer")
            # Retrieves the documents and keep them in the source_documents_key
            | RunnablePassthrough.assign(
                **{
                    # Selects the value to use to retrieve documents from the store
                    self.__source_documents_key: self.__retrieval_query_builder
                    # Retrieves the documents
                    | self.wrap(self.retriever, "Retriever")
                }
            )
            # Builds the answer value by executing the RAG
            | RunnablePassthrough.assign(
                **{
                    self.__answer_key: (
                        # Builds the context variable content
                        RunnablePassthrough.assign(
                            **{
                                self.__param_context: (
                                    # Selects the value to use to retrieve documents from the store
                                    (lambda x: x.get(self.__source_documents_key, []))
                                    # Formats the documents into a single string
                                    | self.wrap(
                                        self.document_formatter, "Document Formatter"
                                    )
                                ),
                            }
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
                }
            )
        )
        if self.__rag_return_sources_formatted_as_string:
            # Formats the source documents as a single string
            chain = chain | self.wrap(
                self.final_answer_formatter, "Final Answer Formatter"
            )
        # INFO: uses a cast to avoid LSP error about incompatible types
        return cast(Runnable[ChainInputType, ChainOutputVar], chain)

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
        if self.__use_rag:
            if self.__rag_return_sources:
                return self.__build_with_rag_with_sources()
            return self.__build_with_rag()
        return self.__build_without_rag()
