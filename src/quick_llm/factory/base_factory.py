"""
Defines the BaseFactory class for creating language model instances with
customizable input/output transformations and logging.
"""

import importlib.util
import logging
from abc import ABC, abstractmethod
from typing import AsyncIterator, Callable, Generic, Iterator, Self, overload

from langchain.chat_models import init_chat_model
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.prompts.string import PromptTemplateFormat
from langchain_core.retrievers import RetrieverLike
from langchain_core.runnables import Runnable, RunnableGenerator
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TextSplitter,
    TokenTextSplitter,
)
from pydantic import BaseModel

from ..support import ChainInputType, ChainOutputVar, ModelTypeVar, RagDocumentIngestor


# pylint: disable=too-many-instance-attributes, too-many-public-methods
class BaseFactory(ABC, Generic[ChainOutputVar, ModelTypeVar]):
    """
    Abstract base class for building language model instances with customizable
    input/output transformations and logging.
    """

    def __init__(
        self,
        output_type: type[ChainOutputVar] = str,
    ) -> None:
        self._output_type = output_type
        # Logger setup
        self._logger = logging.getLogger(__name__)
        self._detailed_logging: bool = False
        # LLM components
        self._language_model: LanguageModelLike | None = None
        self._prompt_template: BasePromptTemplate[PromptValue] | None = None
        # RAG components
        self._text_splitter: TextSplitter | None = None
        self._embeddings: Embeddings | None = None
        self._vector_store: VectorStore | None = None
        self._retriever: RetrieverLike | None = None

    def _fail(self, message: str) -> Exception:
        self._logger.error(message)
        return RuntimeError(message)

    @staticmethod
    def get_readable_value(value: object) -> object:
        """
        Converts the input object into a human-readable format.

        :param value: The object to be converted. This can be a BaseMessage,
        BaseModel, or other types.
        :return: A human-readable representation of the object.
        """
        # WARN: If there are non-serializable objects, this method should be
        # updated to handle them or it will fail
        if isinstance(value, BaseMessage):
            return value.model_dump_json(indent=2)
        if isinstance(value, BaseModel):
            return value.model_dump_json(indent=2)
        # elif isinstance(value, dict):
        #     return json.dumps(value, indent=2)
        return value

    def passthrough_logger[T](self, caption: str) -> Runnable[T, T]:
        """
        Captures the outputs and logs it. It is included in the default
        implementation of `wrap_chain` method.
        """

        def output_collector(output: Iterator[T]) -> Iterator[T]:
            for item in output:
                self._logger.debug(f"{caption}: %s", self.get_readable_value(item))
                yield item

        async def aoutput_collector(output: AsyncIterator[T]) -> AsyncIterator[T]:
            async for item in output:
                self._logger.debug(f"{caption}: %s", self.get_readable_value(item))
                yield item

        return RunnableGenerator(
            output_collector, aoutput_collector, name=f"{caption} output Logger"
        )

    def wrap[Input, Output](
        self, runnable: Runnable[Input, Output], caption: str
    ) -> Runnable[Input, Output]:
        """
        Wraps a runnable with detailed logging if enabled.

        :param runnable: The runnable to be wrapped.
        :return: The wrapped runnable with logging if detailed logging is enabled.
        """
        if self._detailed_logging:
            return runnable | self.passthrough_logger(caption)
        return runnable

    @property
    def language_model(self) -> LanguageModelLike:
        """
        Gets the language model instance.

        :return: The current instance of BaseLanguageModel or None if not set.
        """
        if self._language_model is None:
            raise self._fail("Language model is not set.")
        return self._language_model

    @property
    def prompt_template(self) -> BasePromptTemplate[PromptValue]:
        """
        Gets the prompt template instance.

        :return: The current instance of PromptTemplate or None if not set.
        """
        if self._prompt_template is None:
            raise self._fail("Prompt template is not set.")
        return self._prompt_template

    @property
    def text_splitter(self) -> TextSplitter:
        """
        Gets the text splitter instance.

        :return: The current instance of TextSplitter.
        """
        if self._text_splitter is None:
            raise self._fail("Text splitter is not set.")
        return self._text_splitter

    @property
    def embeddings(self) -> Embeddings:
        """
        Gets the embeddings instance.

        :return: The current instance of Embeddings.
        """
        if self._embeddings is None:
            raise self._fail("Embeddings are not set.")
        return self._embeddings

    @property
    def vector_store(self) -> VectorStore:
        """
        Gets the vector store instance.

        :return: The current instance of VectorStore.
        """
        if self._vector_store is None:
            raise self._fail("Vector store is not set.")
        return self._vector_store

    @property
    def retriever(self) -> RetrieverLike:
        """
        Gets the retriever instance.

        :return: The current instance of RetrieverLike.
        """
        if self._retriever is None:
            raise self._fail("Retriever is not set.")
        return self._retriever

    @property
    def ingestor(self) -> RagDocumentIngestor:
        """
        Creates and returns an instance of RagDocumentIngestor.

        This method initializes a RagDocumentIngestor using the currently set vector
        store and text splitter. These components must be configured
        prior to calling this method, otherwise, an error will be raised.

        :return: A configured RagDocumentIngestor instance.
        :raises RuntimeError: If either vector store or text splitter is not set.
        """
        self._logger.debug("Creating RagDocumentIngestor")
        if not self._vector_store or not self.text_splitter:
            raise self._fail(
                "Cannot create RagDocumentIngestor without vector store and text splitter."
            )
        return RagDocumentIngestor(
            vector_store=self.vector_store,
            text_splitter=self.text_splitter,
        )

    def use(self, visitor: Callable[[Self], None]) -> Self:
        """
        Applies a visitor function to the ChainFactory instance.

        :param visitor: A callable that takes a ChainFactory instance and returns None.
        :return: The ChainFactory instance for method chaining.
        """
        self._logger.debug("Applying visitor to ChainFactory")
        visitor(self)
        return self

    def use_detailed_logging(self, enable: bool = True) -> Self:
        """
        Enables or disables detailed logging for the ChainFactory.

        :param enable: A boolean flag to enable or disable detailed logging. Defaults to True.
        :return: The ChainFactory instance for method chaining.
        """
        self._detailed_logging = enable
        self._logger.debug("Setting detailed logging to %s", self._detailed_logging)
        return self

    @overload
    def use_language_model(self, model: LanguageModelLike) -> Self:
        pass

    @overload
    def use_language_model(self, model: str, **kwargs) -> Self:
        pass

    def use_language_model(
        self,
        model: LanguageModelLike | str,
        **kwargs,
    ) -> Self:
        """
        Sets the language model instance.

        :param language_model: An instance of BaseLanguageModel to set.
        :return: The ChainFactory instance for method chaining.
        """
        if isinstance(model, str):
            self._language_model = init_chat_model(model, **kwargs)
        else:
            self._language_model = model

        self._logger.debug("Setting language model: %s", self._language_model)
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
        self._prompt_template = prompt_template
        return self

    def use_text_splitter(self, text_splitter: TextSplitter) -> Self:
        """
        Sets the text splitter instance.

        :param text_splitter: An instance of TextSplitter to set.
        :return: The ChainFactory instance for method chaining.
        """
        self._text_splitter = text_splitter
        self._logger.debug("Setting text splitter: %s", self._text_splitter)
        return self

    def use_default_token_splitter(
        self, chunk_size: int = 500, chunk_overlap: int = 50
    ) -> Self:
        """
        Sets up a Token TextSplitter with the provided values or the default ones if omitted
        """
        self._text_splitter = TokenTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return self

    def use_default_text_splitter(
        self, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> Self:
        """
        Sets up a Recursive TextSplitter with the provided values or the default ones if omitted
        """
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return self

    def use_embeddings(self, embeddings: Embeddings) -> Self:
        """
        Sets the embeddings instance.

        :param embeddings: An instance of Embeddings to set.
        :return: The ChainFactory instance for method chaining.
        """
        self._embeddings = embeddings
        self._logger.debug("Setting embeddings: %s", self._embeddings)
        return self

    def use_vector_store(self, vector_store: VectorStore) -> Self:
        """
        Sets the vector store instance and enables Retrieval-Augmented Generation (RAG).

        By default, the vector store is also used as a retriever.

        :param vector_store: An instance of VectorStore to set.
        :return: The ChainFactory instance for method chaining.
        """
        self._vector_store = vector_store
        self._logger.debug("Setting vector store: %s", self._vector_store)
        # By default, uses the vector store as retriever
        self._retriever = vector_store.as_retriever()
        return self

    @overload
    def use_retriever(self, retriever: RetrieverLike) -> Self:
        """
        Sets the retriever instance.

        :param retriever: An instance of RetrieverLike to set.
        :return: The ChainFactory instance for method chaining.
        """

    @overload
    def use_retriever(
        self,
        retriever: Callable[[LanguageModelLike, RetrieverLike | None], RetrieverLike],
    ) -> Self:
        """
        Sets the retriever instance using a callable builder.
        :param retriever: A callable that takes a LanguageModelLike instance
        and an optional existing retriever to produce a new RetrieverLike
        instance.
        :return: The ChainFactory instance for method chaining.
        """

    def use_retriever(
        self,
        retriever: RetrieverLike
        | Callable[[LanguageModelLike, RetrieverLike | None], RetrieverLike]
        | None = None,
    ) -> Self:
        """
        Sets a custom retriever instance or builds one using the provided callable.

        This method ensures retrieval-augmented generation (RAG) is enabled and
        assigns the retriever provided. If the retriever is given as a
        callable, it evaluates the callable with the current language model and
        the existing retriever (if any) to construct a new retriever.

        :param retriever: Either a `RetrieverLike` instance or a callable that
        takes a `LanguageModelLike` instance and an optional existing retriever
        to produce a new one.
        :return: The ChainFactory instance for method chaining.
        """
        if isinstance(retriever, Callable):
            retriever = retriever(self.language_model, self._retriever)
        self._retriever = retriever
        self._logger.debug("Setting retriever: %s", self._retriever)
        return self

    def _pretty_runnable(self, runnable: Runnable) -> str:
        """
        Generates a pretty string representation of a Runnable chain.

        :param runnable: The Runnable instance to be represented.
        :return: A string representation of the Runnable chain.
        """
        # Check if "grandalf" is installed for ASCII graph drawing
        if importlib.util.find_spec("grandalf") is None:
            self._logger.warning(
                "'grandalf' package not found. Install it to see the chain graph."
            )
            return str(runnable)

        graph = runnable.get_graph()
        return "\n" + graph.draw_ascii()

    @abstractmethod
    def build(
        self,
    ) -> Runnable[ChainInputType, ChainOutputVar]:
        """
        Builds and returns a Runnable that processes input through the language
        model.
        """
