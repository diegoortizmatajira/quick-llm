"""Tests for the ChainFactory class"""

import json
from typing import cast

from langchain_core.prompt_values import ChatPromptValue, StringPromptValue
from langchain_core.prompts.chat import ChatPromptTemplate, MessagePromptTemplateT
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.language_models import (
    FakeListChatModel,
    FakeListLLM,
    FakeStreamingListLLM,
    LanguageModelLike,
    LanguageModelOutput,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.retrievers import RetrieverLike
from langchain_core.runnables import RunnableLambda
from langchain_core.vectorstores import InMemoryVectorStore
from pydantic import BaseModel, Field

from quick_llm import ChainFactory, ChainInputType

TEST_INPUT = "Test input"
TEST_EXPECTED_RESPONSE = "This is a sample response."
TEST_JSON_EXPECTED_RESPONSE = json.dumps(
    {
        "what": "something",
        "when": "tomorrow",
        "who": "someone",
        "general": "something else",
    }
)
TEST_INPUT_SAMPLES: list[ChainInputType] = [
    {"input": TEST_INPUT},
    TEST_INPUT,
]
TEST_ESCAPED_BAD_STRING = r"This is a bad \_ json string"
TEST_ESCAPED_FIXED_STRING = "This is a bad _ json string"
TEST_FAKE_DOCUMENT1 = Document(
    page_content="This is a fake document",
    metadata={"source": "FakeDocument"},
)
TEST_FAKE_DOCUMENT2 = Document(
    page_content="This is another fake document with no metadata",
)
TEST_DOCUMENT_LIST = [TEST_FAKE_DOCUMENT1, TEST_FAKE_DOCUMENT2]
TEST_EXPECTED_CUSTOM_RETRIEVER_RESULT = [TEST_FAKE_DOCUMENT2]


class AnswerOutput(BaseModel):
    """Sample object structure to test the JSON parsing feature"""

    what: str = Field(description="Summarizes/rephrase the question being answered.")
    when: str = Field(
        description="Provides a date-formatted answer to the question when required."
    )
    who: str = Field(
        description="Provides a proper name answer to the question when required."
    )
    general: str = Field(description="Provides a short-text answer to the question.")


def _get_test_models(expected_response: str) -> list[LanguageModelLike]:
    return [
        FakeListLLM(responses=[expected_response]),
        FakeListChatModel(responses=[expected_response]),
    ]


def _get_test_streaming_models(expected_response: str) -> list[LanguageModelLike]:
    return [
        FakeStreamingListLLM(responses=[expected_response]),
        FakeListChatModel(responses=[expected_response]),
    ]


class TestBaseChains:
    """Tests for the ChainFactory class basic functionality"""

    @pytest.mark.parametrize("input_value", TEST_INPUT_SAMPLES)
    @pytest.mark.parametrize("model", _get_test_models(TEST_EXPECTED_RESPONSE))
    def test_string(self, input_value: ChainInputType, model: LanguageModelLike):
        """Test the factory with a string output"""
        factory = (
            ChainFactory()
            .use_prompt_template("Sample Prompt {input}")
            .use_language_model(model)
            .use_detailed_logging()
        )
        chain = factory.build()
        response = chain.invoke(input_value)
        assert response == TEST_EXPECTED_RESPONSE

    @pytest.mark.asyncio
    @pytest.mark.parametrize("input_value", TEST_INPUT_SAMPLES)
    @pytest.mark.parametrize("model", _get_test_models(TEST_EXPECTED_RESPONSE))
    async def test_string_async(
        self, input_value: ChainInputType, model: LanguageModelLike
    ):
        """Test the factory with a simple text chain"""
        factory = (
            ChainFactory()
            .use_prompt_template("Sample Prompt {input}")
            .use_language_model(model)
        )
        chain = factory.build()
        response = await chain.ainvoke(input_value)
        assert response == TEST_EXPECTED_RESPONSE

    @pytest.mark.parametrize("input_value", TEST_INPUT_SAMPLES)
    @pytest.mark.parametrize(
        "model", _get_test_streaming_models(TEST_EXPECTED_RESPONSE)
    )
    def test_string_stream(self, input_value: ChainInputType, model: LanguageModelLike):
        """Test the factory with a simple text chain using streaming on a chat model and a non-chat model"""
        factory = (
            ChainFactory()
            .use_prompt_template("Sample Prompt {input}")
            .use_language_model(model)
        )
        chain = factory.build()
        # Verify streaming response to ensure no chunks are equal to the full response
        stream = [
            item for item in chain.stream(input_value) if item != TEST_EXPECTED_RESPONSE
        ]
        assert len(stream) > 0
        # Reconstruct full response from stream and verify correctness
        response = "".join(stream)
        assert response == TEST_EXPECTED_RESPONSE

    @pytest.mark.parametrize("input_value", TEST_INPUT_SAMPLES)
    @pytest.mark.parametrize("model", _get_test_models(TEST_JSON_EXPECTED_RESPONSE))
    def test_json(self, input_value: ChainInputType, model: LanguageModelLike):
        """Test the factory with a json output"""
        factory = (
            ChainFactory.for_structured_output(AnswerOutput)
            .use_prompt_template("Sample Prompt {input}")
            .use_language_model(model)
            .use_detailed_logging()
        )
        chain = factory.build()
        response = chain.invoke(input_value)
        # Checks that the response is a populated dictionary
        assert isinstance(response, dict)
        assert response.get("what", "nothing") == "something"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("input_value", TEST_INPUT_SAMPLES)
    @pytest.mark.parametrize("model", _get_test_models(TEST_JSON_EXPECTED_RESPONSE))
    async def test_json_async(
        self, input_value: ChainInputType, model: LanguageModelLike
    ):
        """Test the factory with a json output"""
        factory = (
            ChainFactory.for_structured_output(AnswerOutput)
            .use_prompt_template("Sample Prompt {input}")
            .use_language_model(model)
        )
        chain = factory.build()
        response = await chain.ainvoke(input_value)
        # Checks that the response is a populated dictionary
        assert isinstance(response, dict)
        assert response.get("what", "nothing") == "something"

    @pytest.mark.parametrize("input_value", TEST_INPUT_SAMPLES)
    @pytest.mark.parametrize(
        "model", _get_test_streaming_models(TEST_JSON_EXPECTED_RESPONSE)
    )
    def test_json_stream(self, input_value: ChainInputType, model: LanguageModelLike):
        """Test the factory with a json output"""
        factory = (
            ChainFactory.for_structured_output(AnswerOutput)
            .use_prompt_template("Sample Prompt {input}")
            .use_language_model(model)
        )
        chain = factory.build()
        stream = chain.stream(input_value)
        response = list(stream)
        assert len(response) > 0
        # INFO: Not a very practical output.
        assert response[0] == {}
        assert response[1] == {"what": ""}
        assert response[2] == {"what": "s"}
        assert response[3] == {"what": "so"}
        assert response[4] == {"what": "som"}
        assert response[5] == {"what": "some"}
        assert response[6] == {"what": "somet"}
        assert response[7] == {"what": "someth"}
        assert response[8] == {"what": "somethi"}
        assert response[9] == {"what": "somethin"}
        assert response[10] == {"what": "something"}
        assert response[11] == {"what": "something", "when": ""}
        assert response[12] == {"what": "something", "when": "t"}
        assert response[13] == {"what": "something", "when": "to"}
        assert response[14] == {"what": "something", "when": "tom"}
        assert response[15] == {"what": "something", "when": "tomo"}
        assert response[16] == {"what": "something", "when": "tomor"}
        assert response[17] == {"what": "something", "when": "tomorr"}
        assert response[18] == {"what": "something", "when": "tomorro"}
        assert response[19] == {"what": "something", "when": "tomorrow"}
        assert response[20] == {"what": "something", "when": "tomorrow", "who": ""}


class TestBaseSupportComponents:
    """Tests for the ChainFactory class support components"""

    def test_string_prompt(self):
        """Tests the prompt template component of the ChainFactory"""
        factory = ChainFactory().use_prompt_template("Sample Prompt {input}")
        prompt = factory.prompt_template
        assert prompt is not None
        rendered = prompt.invoke(
            {
                factory.input_param: TEST_INPUT,
            }
        )
        assert isinstance(rendered, StringPromptValue)
        assert rendered.text == f"Sample Prompt {TEST_INPUT}"

    def test_custom_prompt(self):
        """Tests the prompt template component of the ChainFactory"""
        factory = (
            ChainFactory()
            .use_prompt_template(
                ChatPromptTemplate(
                    [
                        ("system", "You are a helpful assistant."),
                        ("human", "Sample Prompt {input}"),
                    ],
                    template_format="f-string",
                )
            )
            .use_input_param("input")
        )
        prompt = factory.prompt_template
        assert prompt is not None
        assert isinstance(prompt, ChatPromptTemplate)
        rendered = prompt.invoke(
            {
                factory.input_param: TEST_INPUT,
            }
        )
        assert isinstance(rendered, ChatPromptValue)
        assert len(rendered.messages) == 2
        assert isinstance(rendered.messages[0], SystemMessage)
        assert rendered.messages[0].content == "You are a helpful assistant."
        assert isinstance(rendered.messages[1], HumanMessage)
        assert rendered.messages[1].content == f"Sample Prompt {TEST_INPUT}"

    @pytest.mark.parametrize("input_value", TEST_INPUT_SAMPLES)
    def test_input_transformer(self, input_value: ChainInputType):
        """Tests the input parser to ensure it accepts various input formats"""
        factory = ChainFactory()
        transformer = factory.input_transformer
        output = transformer.invoke(input_value)
        assert output is not None
        assert output.get(factory.input_param, "empty") == TEST_INPUT

    def test_additional_values_injector(self):
        """Test that additional values are correctly injected for the JSON model.

        Verifies that the ChainFactory's additional_values_injector adds the necessary
        fields to the input, preserves the original input, and injects specific fields
        required for a JSON model.
        """
        factory = ChainFactory.for_structured_output(AnswerOutput)
        injector = factory.additional_values_injector
        result = injector.invoke({factory.input_param: TEST_INPUT})
        assert result is not None
        # Check that the original input is preserved
        assert result.get(factory.input_param, "empty") == TEST_INPUT
        # As the factory is for a JSON model, format_instructions should be injected
        assert result.get(factory.format_instructions_param, "empty") != "empty"

    @pytest.mark.parametrize(
        "test_input,expected_output",
        [
            (TEST_ESCAPED_BAD_STRING, TEST_ESCAPED_FIXED_STRING),
            (
                BaseMessage(content=[TEST_ESCAPED_BAD_STRING], type="test"),
                BaseMessage(content=[TEST_ESCAPED_FIXED_STRING], type="test"),
            ),
            (
                BaseMessage(content=TEST_ESCAPED_BAD_STRING, type="test"),
                BaseMessage(content=TEST_ESCAPED_FIXED_STRING, type="test"),
            ),
            (
                AIMessage(content=[TEST_ESCAPED_BAD_STRING]),
                AIMessage(content=[TEST_ESCAPED_FIXED_STRING]),
            ),
            (
                AIMessage(content=TEST_ESCAPED_BAD_STRING),
                AIMessage(content=TEST_ESCAPED_FIXED_STRING),
            ),
        ],
    )
    def test_output_cleaner(
        self, test_input: LanguageModelOutput, expected_output: LanguageModelOutput
    ):
        """Test the text cleaning function"""
        llm = ChainFactory()
        cleaner = llm.output_cleaner
        output = cleaner.invoke(test_input)
        assert output == expected_output


def _rag_setup_visitor(model: LanguageModelLike):
    """Set up a visitor to configure a ChainFactory for Retrieval-Augmented Generation (RAG).

    This function returns a visitor that can configure a ChainFactory with necessary components
    for RAG testing, including setting up prompt templates, language models, text splitters,
    embeddings, vector stores, retrievers, and detailed logging.

    Args:
        model (LanguageModelLike): The language model used in the RAG setup.

    Returns:
        Callable[[ChainFactory], None]: A visitor function that modifies a ChainFactory instance.
    """

    def visitor(factory: ChainFactory):
        """Configures a ChainFactory for RAG testing"""

        mock_embeddings = FakeEmbeddings(size=3)
        mock_vectorstore = InMemoryVectorStore(mock_embeddings)
        mock_vectorstore.add_documents(TEST_DOCUMENT_LIST)
        _ = (
            factory.use_prompt_template("Sample Prompt {input} {context}")
            .use_language_model(model)
            .use_default_text_splitter()
            .use_embeddings(mock_embeddings)
            .use_vector_store(mock_vectorstore)
            .use_detailed_logging()
        )

    return visitor


class TestRagChains:
    """Tests for the ChainFactory class RAG functionality"""

    @pytest.mark.parametrize("input_value", TEST_INPUT_SAMPLES)
    @pytest.mark.parametrize("model", _get_test_models(TEST_EXPECTED_RESPONSE))
    def test_rag_string(self, input_value: ChainInputType, model: LanguageModelLike):
        """Test the factory with a string output"""
        factory = ChainFactory().use(_rag_setup_visitor(model))
        chain = factory.build()
        response = chain.invoke(input_value)
        assert response == TEST_EXPECTED_RESPONSE

    @pytest.mark.parametrize("input_value", TEST_INPUT_SAMPLES)
    @pytest.mark.parametrize("model", _get_test_models(TEST_EXPECTED_RESPONSE))
    async def test_rag_string_async(
        self, input_value: ChainInputType, model: LanguageModelLike
    ):
        """Test the factory with a string output"""
        factory = ChainFactory().use(_rag_setup_visitor(model))
        chain = factory.build()
        response = await chain.ainvoke(input_value)
        assert response == TEST_EXPECTED_RESPONSE

    @pytest.mark.parametrize("input_value", TEST_INPUT_SAMPLES)
    @pytest.mark.parametrize(
        "model", _get_test_streaming_models(TEST_EXPECTED_RESPONSE)
    )
    def test_rag_string_stream(
        self, input_value: ChainInputType, model: LanguageModelLike
    ):
        """Test the factory with a simple text chain using streaming on a chat model and a non-chat model"""
        factory = ChainFactory().use(_rag_setup_visitor(model))
        chain = factory.build()
        # Verify streaming response to ensure no chunks are equal to the full response
        stream = [
            item for item in chain.stream(input_value) if item != TEST_EXPECTED_RESPONSE
        ]
        assert len(stream) > 0
        # Reconstruct full response from stream and verify correctness
        response = "".join(stream)
        assert response == TEST_EXPECTED_RESPONSE

    @pytest.mark.parametrize("input_value", TEST_INPUT_SAMPLES)
    @pytest.mark.parametrize("model", _get_test_models(TEST_EXPECTED_RESPONSE))
    def test_rag_string_with_documents(
        self, input_value: ChainInputType, model: LanguageModelLike
    ):
        """Test the factory with a string output"""
        factory = (
            ChainFactory()
            .use(_rag_setup_visitor(model))
            .use_rag_returning_sources(True, True)
        )
        chain = factory.build()
        response = chain.invoke(input_value)
        assert isinstance(response, str)
        assert response.startswith(TEST_EXPECTED_RESPONSE)
        # Length should be greater than expected response due to appended sources
        assert len(response) > len(TEST_EXPECTED_RESPONSE)

    @pytest.mark.parametrize("input_value", TEST_INPUT_SAMPLES)
    @pytest.mark.parametrize("model", _get_test_models(TEST_EXPECTED_RESPONSE))
    async def test_rag_string_with_documents_async(
        self, input_value: ChainInputType, model: LanguageModelLike
    ):
        """Test the factory with a string output"""
        factory = (
            ChainFactory()
            .use(_rag_setup_visitor(model))
            .use_rag_returning_sources(True, True)
        )
        chain = factory.build()
        response = await chain.ainvoke(input_value)
        assert isinstance(response, str)
        assert response.startswith(TEST_EXPECTED_RESPONSE)
        # Length should be greater than expected response due to appended sources
        assert len(response) > len(TEST_EXPECTED_RESPONSE)

    @pytest.mark.parametrize("input_value", TEST_INPUT_SAMPLES)
    @pytest.mark.parametrize(
        "model", _get_test_streaming_models(TEST_EXPECTED_RESPONSE)
    )
    def test_rag_string_with_documents_stream(
        self, input_value: ChainInputType, model: LanguageModelLike
    ):
        """Test the factory with a simple text chain using streaming on a chat model and a non-chat model"""
        factory = (
            ChainFactory()
            .use(_rag_setup_visitor(model))
            .use_rag_returning_sources(True, True)
        )
        chain = factory.build()
        # Verify streaming response to ensure no chunks are equal to the full response
        stream = [
            item for item in chain.stream(input_value) if item != TEST_EXPECTED_RESPONSE
        ]
        assert len(stream) > 0
        # Reconstruct full response from stream and verify correctness
        response = "".join(stream)
        assert response.startswith(TEST_EXPECTED_RESPONSE)
        # Length should be greater than expected response due to appended sources
        assert len(response) > len(TEST_EXPECTED_RESPONSE)

    @pytest.mark.parametrize("input_value", TEST_INPUT_SAMPLES)
    @pytest.mark.parametrize("model", _get_test_models(TEST_EXPECTED_RESPONSE))
    def test_rag_dict_with_documents(
        self, input_value: ChainInputType, model: LanguageModelLike
    ):
        """Test the factory with a string output"""
        factory = ChainFactory.for_rag_with_sources().use(_rag_setup_visitor(model))
        chain = factory.build()
        response = chain.invoke(input_value)
        assert isinstance(response, dict)
        assert response.get(factory.answer_key, "None") == TEST_EXPECTED_RESPONSE
        referenced_docs = cast(list, response.get(factory.document_references_key, []))
        assert len(referenced_docs) == len(TEST_DOCUMENT_LIST)

    @pytest.mark.parametrize("input_value", TEST_INPUT_SAMPLES)
    @pytest.mark.parametrize("model", _get_test_models(TEST_EXPECTED_RESPONSE))
    async def test_rag_dict_with_documents_async(
        self, input_value: ChainInputType, model: LanguageModelLike
    ):
        """Test the factory with a string output"""
        factory = ChainFactory.for_rag_with_sources().use(_rag_setup_visitor(model))
        chain = factory.build()
        response = await chain.ainvoke(input_value)
        assert isinstance(response, dict)
        assert response.get(factory.answer_key, "None") == TEST_EXPECTED_RESPONSE
        referenced_docs = cast(list, response.get(factory.document_references_key, []))
        assert len(referenced_docs) == len(TEST_DOCUMENT_LIST)

    @pytest.mark.parametrize("input_value", TEST_INPUT_SAMPLES)
    @pytest.mark.parametrize(
        "model", _get_test_streaming_models(TEST_EXPECTED_RESPONSE)
    )
    def test_rag_dict_with_documents_stream(
        self, input_value: ChainInputType, model: LanguageModelLike
    ):
        """Test the factory with a simple text chain using streaming on a chat model and a non-chat model"""
        factory = ChainFactory.for_rag_with_sources().use(_rag_setup_visitor(model))
        chain = factory.build()
        # Verify streaming response to ensure no chunks are equal to the full response
        stream = [
            item for item in chain.stream(input_value) if item != TEST_EXPECTED_RESPONSE
        ]
        assert len(stream) > 0
        # Reconstruct full response from stream and verify correctness
        text_response = ""
        referenced_docs = []
        for chunk in stream:
            text_response += str(chunk.get(factory.answer_key, ""))
            if factory.document_references_key in chunk:
                referenced_docs = cast(list, chunk[factory.document_references_key])

        assert text_response == TEST_EXPECTED_RESPONSE
        assert len(referenced_docs) == len(TEST_DOCUMENT_LIST)

    @pytest.mark.parametrize("input_value", TEST_INPUT_SAMPLES)
    @pytest.mark.parametrize("model", _get_test_models(TEST_JSON_EXPECTED_RESPONSE))
    def test_rag_json(self, input_value: ChainInputType, model: LanguageModelLike):
        """Test the factory with a json output"""
        factory = ChainFactory.for_structured_output(AnswerOutput).use(
            _rag_setup_visitor(model)
        )
        chain = factory.build()
        response = chain.invoke(input_value)
        # Checks that the response is a populated dictionary
        assert isinstance(response, dict)
        assert response.get("what", "nothing") == "something"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("input_value", TEST_INPUT_SAMPLES)
    @pytest.mark.parametrize("model", _get_test_models(TEST_JSON_EXPECTED_RESPONSE))
    async def test_rag_json_async(
        self, input_value: ChainInputType, model: LanguageModelLike
    ):
        """Test the factory with a json output"""
        factory = ChainFactory.for_structured_output(AnswerOutput).use(
            _rag_setup_visitor(model)
        )
        chain = factory.build()
        response = await chain.ainvoke(input_value)
        # Checks that the response is a populated dictionary
        assert isinstance(response, dict)
        assert response.get("what", "nothing") == "something"

    @pytest.mark.parametrize("input_value", TEST_INPUT_SAMPLES)
    @pytest.mark.parametrize(
        "model", _get_test_streaming_models(TEST_JSON_EXPECTED_RESPONSE)
    )
    def test_rag_json_stream(
        self, input_value: ChainInputType, model: LanguageModelLike
    ):
        """Test the factory with a json output"""
        factory = ChainFactory.for_structured_output(AnswerOutput).use(
            _rag_setup_visitor(model)
        )
        chain = factory.build()
        stream = chain.stream(input_value)
        response = list(stream)
        assert len(response) > 0
        # INFO: Not a very practical output.
        assert response[0] == {}
        assert response[1] == {"what": ""}
        assert response[2] == {"what": "s"}
        assert response[3] == {"what": "so"}
        assert response[4] == {"what": "som"}
        assert response[5] == {"what": "some"}
        assert response[6] == {"what": "somet"}
        assert response[7] == {"what": "someth"}
        assert response[8] == {"what": "somethi"}
        assert response[9] == {"what": "somethin"}
        assert response[10] == {"what": "something"}
        assert response[11] == {"what": "something", "when": ""}
        assert response[12] == {"what": "something", "when": "t"}
        assert response[13] == {"what": "something", "when": "to"}
        assert response[14] == {"what": "something", "when": "tom"}
        assert response[15] == {"what": "something", "when": "tomo"}
        assert response[16] == {"what": "something", "when": "tomor"}
        assert response[17] == {"what": "something", "when": "tomorr"}
        assert response[18] == {"what": "something", "when": "tomorro"}
        assert response[19] == {"what": "something", "when": "tomorrow"}
        assert response[20] == {"what": "something", "when": "tomorrow", "who": ""}


class TestRagSupportComponents:
    """Tests for the ChainFactory class RAG support components"""

    @pytest.mark.parametrize("input_value", TEST_INPUT_SAMPLES)
    def test_use_retriever(self, input_value: ChainInputType):
        """Test the factory with a string output"""
        models = _get_test_models(TEST_EXPECTED_RESPONSE)
        mock_retriever: RetrieverLike = RunnableLambda(
            lambda _: TEST_EXPECTED_CUSTOM_RETRIEVER_RESULT
        )
        for model in models:
            factory = (
                ChainFactory()
                .use(_rag_setup_visitor(model))
                .use_retriever(mock_retriever)
            )
            retrieve_result = factory.retriever.invoke("test query")
            assert retrieve_result == TEST_EXPECTED_CUSTOM_RETRIEVER_RESULT
            chain = factory.build()
            response = chain.invoke(input_value)
            assert response == TEST_EXPECTED_RESPONSE


class TestParameterConfiguration:
    """Test custom parameter name configuration"""

    def test_use_input_param(self):
        """Test customizing the input parameter name"""
        factory = (
            ChainFactory()
            .use_input_param("query")
            .use_prompt_template("Answer this {query}")
            .use_language_model(FakeListLLM(responses=[TEST_EXPECTED_RESPONSE]))
        )

        assert factory.input_param == "query"

        chain = factory.build()
        # Should work with custom parameter name
        response = chain.invoke({"query": TEST_INPUT})
        assert response == TEST_EXPECTED_RESPONSE

    def test_use_format_instructions_param(self):
        """Test customizing the format_instructions parameter name"""
        factory = (
            ChainFactory.for_structured_output(AnswerOutput)
            .use_format_instructions_param("instructions")
            .use_prompt_template("Answer {input}\n\n{instructions}")
            .use_language_model(FakeListLLM(responses=['{"answer": "test"}']))
        )

        assert factory.format_instructions_param == "instructions"

        # Verify format instructions are injected with custom name
        injector = factory.additional_values_injector
        result = injector.invoke({"input": TEST_INPUT})
        assert "instructions" in result
        assert result["instructions"] != ""

    def test_use_context_param(self):
        """Test customizing the context parameter name for RAG"""
        mock_embeddings = FakeEmbeddings(size=3)
        mock_vectorstore = InMemoryVectorStore(mock_embeddings)
        mock_vectorstore.add_documents(TEST_DOCUMENT_LIST)

        factory = (
            ChainFactory()
            .use_context_param("retrieved_context")
            .use_prompt_template(
                "Question: {input}\n\nContext: {retrieved_context}\n\nAnswer:"
            )
            .use_language_model(FakeListLLM(responses=[TEST_EXPECTED_RESPONSE]))
            .use_embeddings(mock_embeddings)
            .use_vector_store(mock_vectorstore)
        )

        chain = factory.build()
        response = chain.invoke(TEST_INPUT)
        assert response == TEST_EXPECTED_RESPONSE

    def test_use_answer_key(self):
        """Test customizing the answer key for RAG with sources"""
        mock_embeddings = FakeEmbeddings(size=3)
        mock_vectorstore = InMemoryVectorStore(mock_embeddings)
        mock_vectorstore.add_documents(TEST_DOCUMENT_LIST)

        factory = (
            ChainFactory.for_rag_with_sources()
            .use_answer_key("response")
            .use_prompt_template("Q: {input}\nContext: {context}\nA:")
            .use_language_model(FakeListLLM(responses=[TEST_EXPECTED_RESPONSE]))
            .use_embeddings(mock_embeddings)
            .use_vector_store(mock_vectorstore)
        )

        assert factory.answer_key == "response"

        chain = factory.build()
        result = chain.invoke(TEST_INPUT)
        assert isinstance(result, dict)
        assert "response" in result
        assert result["response"] == TEST_EXPECTED_RESPONSE


class TestCustomTransformers:
    """Test custom transformer functions"""

    def test_use_output_transformer(self):
        """Test custom output transformer"""

        def custom_transformer(output: LanguageModelOutput) -> str:
            return f"TRANSFORMED: {cast(str, output).upper()}"

        factory = (
            ChainFactory()
            .use_prompt_template("Answer: {input}")
            .use_language_model(FakeListLLM(responses=[TEST_EXPECTED_RESPONSE]))
            .use_output_transformer(custom_transformer)
        )

        chain = factory.build()
        response = chain.invoke(TEST_INPUT)
        assert response == f"TRANSFORMED: {TEST_EXPECTED_RESPONSE.upper()}"

    def test_use_custom_output_cleaner(self):
        """Test custom output cleaner function"""

        def custom_cleaner(text: str) -> str:
            return text.replace("bad", "good")

        factory = (
            ChainFactory()
            .use_prompt_template("Say {input}")
            .use_language_model(FakeListLLM(responses=["This is bad text"]))
            .use_custom_output_cleaner(custom_cleaner)
        )

        chain = factory.build()
        response = chain.invoke(TEST_INPUT)
        assert response == "This is good text"

    def test_use_custom_context_formatter(self):
        """Test custom context formatter for RAG"""

        def custom_formatter(documents: list[Document]) -> str:
            return "SOURCES: " + " | ".join(doc.page_content for doc in documents)

        mock_embeddings = FakeEmbeddings(size=3)
        mock_vectorstore = InMemoryVectorStore(mock_embeddings)
        mock_vectorstore.add_documents(
            [
                Document(page_content="Doc1", metadata={}),
                Document(page_content="Doc2", metadata={}),
            ]
        )

        factory = (
            ChainFactory()
            .use_prompt_template("Q: {input}\nContext: {context}\nA:")
            .use_language_model(FakeListLLM(responses=[TEST_EXPECTED_RESPONSE]))
            .use_embeddings(mock_embeddings)
            .use_vector_store(mock_vectorstore)
            .use_custom_context_formatter(custom_formatter)
        )

        # Access document_formatter to verify custom formatter is used
        formatter = factory.document_formatter
        test_docs = [Document(page_content="Test")]
        formatted = formatter.invoke(test_docs)
        assert formatted.startswith("SOURCES:")

    def test_use_custom_retrieval_query_builder(self):
        """Test custom retrieval query builder"""

        def custom_query_builder(input_dict: dict) -> str:
            # Extract and transform the input for retrieval
            return f"SEARCH: {input_dict.get('input', '')}"

        mock_embeddings = FakeEmbeddings(size=3)
        mock_vectorstore = InMemoryVectorStore(mock_embeddings)
        mock_vectorstore.add_documents(TEST_DOCUMENT_LIST)

        factory = (
            ChainFactory()
            .use_prompt_template("Q: {input}\nContext: {context}\nA:")
            .use_language_model(FakeListLLM(responses=[TEST_EXPECTED_RESPONSE]))
            .use_embeddings(mock_embeddings)
            .use_vector_store(mock_vectorstore)
            .use_custom_retrieval_query_builder(custom_query_builder)
        )

        chain = factory.build()
        response = chain.invoke(TEST_INPUT)
        assert response == TEST_EXPECTED_RESPONSE


class TestTextSplitterConfiguration:
    """Test text splitter configuration options"""

    def test_use_text_splitter_custom(self):
        """Test using a custom text splitter"""
        custom_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100, chunk_overlap=20, separators=["\n\n", "\n", " "]
        )

        factory = ChainFactory().use_text_splitter(custom_splitter)

        assert factory.text_splitter == custom_splitter
        assert factory.text_splitter._chunk_size == 100  # type: ignore

    @pytest.mark.skip(
        reason="TokenTextSplitter requires tiktoken which may not be installed"
    )
    def test_use_default_token_splitter(self):
        """Test using the default token splitter"""
        factory = ChainFactory().use_default_token_splitter()

        assert factory.text_splitter is not None
        assert isinstance(factory.text_splitter, TokenTextSplitter)

    def test_use_default_text_splitter(self):
        """Test using the default text splitter"""
        factory = ChainFactory().use_default_text_splitter()

        assert factory.text_splitter is not None
        assert isinstance(factory.text_splitter, RecursiveCharacterTextSplitter)


class TestIngestor:
    """Test RAG document ingestor"""

    def test_ingestor_property(self):
        """Test accessing the RAG document ingestor"""
        mock_embeddings = FakeEmbeddings(size=3)
        mock_vectorstore = InMemoryVectorStore(mock_embeddings)

        factory = (
            ChainFactory()
            .use_embeddings(mock_embeddings)
            .use_vector_store(mock_vectorstore)
            .use_default_text_splitter()
        )

        ingestor = factory.ingestor
        assert ingestor is not None

        # Test ingesting documents
        test_doc = Document(page_content="Ingestor test content")
        ingestor.from_documents([test_doc], use_splitter=False)

        # Verify document was added to vector store
        results = mock_vectorstore.similarity_search("Ingestor", k=1)
        assert len(results) > 0

    def test_ingestor_without_vector_store_fails(self):
        """Test that ingestor fails without vector store"""
        factory = ChainFactory()

        with pytest.raises(RuntimeError, match="Cannot create RagDocumentIngestor"):
            _ = factory.ingestor


class TestErrorHandling:
    """Test error handling and validation"""

    def test_build_without_language_model_fails(self):
        """Test that building without language model fails"""
        factory = ChainFactory().use_prompt_template("Test {input}")

        with pytest.raises(RuntimeError, match="Language model is not set"):
            factory.build()

    def test_build_without_prompt_template_fails(self):
        """Test that building without prompt template fails"""
        factory = ChainFactory().use_language_model(
            FakeListLLM(responses=[TEST_EXPECTED_RESPONSE])
        )

        with pytest.raises(RuntimeError, match="Prompt template is not set"):
            factory.build()

    def test_rag_accessing_retriever_without_setting_fails(self):
        """Test that accessing retriever without setting it raises error"""
        factory = ChainFactory()

        # Accessing retriever without setting vector store or retriever raises error
        with pytest.raises(RuntimeError, match="Retriever is not set"):
            _ = factory.retriever

    def test_rag_with_vector_store_without_embeddings_fails(self):
        """Test that RAG with vector store but without embeddings fails"""
        mock_vectorstore = InMemoryVectorStore(FakeEmbeddings(size=3))

        factory = (
            ChainFactory()
            .use_prompt_template("Q: {input}\nContext: {context}\nA:")
            .use_language_model(FakeListLLM(responses=[TEST_EXPECTED_RESPONSE]))
            .use_vector_store(mock_vectorstore)
        )

        # Accessing embeddings without setting it will raise error
        with pytest.raises(RuntimeError, match="Embeddings are not set"):
            _ = factory.embeddings


class TestStaticFactoryMethods:
    """Test static factory methods"""

    def test_for_json_model_creates_correct_factory(self):
        """Test that for_json_model creates properly configured factory"""
        factory = ChainFactory.for_structured_output(AnswerOutput)

        # Verify output type is dict
        assert factory.output_transformer is not None

        # Verify format instructions are generated
        injector = factory.additional_values_injector
        result = injector.invoke({"input": TEST_INPUT})
        assert factory.format_instructions_param in result


class TestDefaultFormatters:
    """Test default formatter functions"""

    def test_default_cleaner_function(self):
        """Test the default output cleaner"""
        factory = ChainFactory()
        cleaner = factory.default_cleaner_function

        # Should replace escaped underscores
        assert cleaner(r"test \_ string") == "test _ string"
        assert cleaner("normal string") == "normal string"

    def test_default_context_formatter(self):
        """Test the default context formatter"""
        factory = ChainFactory()
        formatter = factory.default_context_formatter

        docs = [
            Document(page_content="First doc"),
            Document(page_content="Second doc"),
        ]

        result = formatter(docs)
        assert "First doc" in result
        assert "Second doc" in result
        assert "\n\n" in result

    def test_default_references_formatter(self):
        """Test the default references formatter"""
        factory = ChainFactory()
        formatter = factory.default_references_formatter

        docs = [
            Document(page_content="Doc1", metadata={"source": "file1.txt"}),
            Document(page_content="Doc2", metadata={"source": "file2.txt"}),
        ]

        result = formatter(docs)
        assert "References:" in result
        assert "file1.txt" in result
        assert "file2.txt" in result
        assert "**[1]**" in result
        assert "**[2]**" in result

    def test_default_references_formatter_without_source(self):
        """Test references formatter with documents lacking source metadata"""
        factory = ChainFactory()
        formatter = factory.default_references_formatter

        docs = [
            Document(page_content="Content without source", metadata={}),
        ]

        result = formatter(docs)
        assert "References:" in result
        assert "Content without source" in result


class TestGetReadableValue:
    """Test get_readable_value static method"""

    def test_get_readable_value_with_base_message(self):
        """Test readable value conversion for BaseMessage"""

        msg = HumanMessage(content="Test message")
        result = ChainFactory.get_readable_value(msg)

        assert isinstance(result, str)
        assert "Test message" in result

    def test_get_readable_value_with_base_model(self):
        """Test readable value conversion for BaseModel"""
        model = AnswerOutput(
            what="test answer", when="now", who="tester", general="general info"
        )
        result = ChainFactory.get_readable_value(model)

        assert isinstance(result, str)
        assert "test answer" in result

    def test_get_readable_value_with_string(self):
        """Test readable value conversion for plain string"""
        result = ChainFactory.get_readable_value("plain text")
        assert result == "plain text"

    def test_get_readable_value_with_dict(self):
        """Test readable value conversion for dict"""
        test_dict = {"key": "value"}
        result = ChainFactory.get_readable_value(test_dict)
        assert result == test_dict


class TestVisitorPattern:
    """Test the visitor pattern functionality"""

    def test_use_method_with_visitor(self):
        """Test using a visitor to configure the factory"""
        visited = False

        def visitor(factory: ChainFactory):
            nonlocal visited
            visited = True
            factory.use_language_model(FakeListLLM(responses=[TEST_EXPECTED_RESPONSE]))
            factory.use_prompt_template("Test {input}")

        factory = ChainFactory().use(visitor)

        assert visited
        assert factory.language_model is not None
        assert factory.prompt_template is not None

        chain = factory.build()
        response = chain.invoke(TEST_INPUT)
        assert response == TEST_EXPECTED_RESPONSE

    def test_multiple_visitors(self):
        """Test chaining multiple visitors"""

        def visitor1(factory: ChainFactory):
            factory.use_language_model(FakeListLLM(responses=[TEST_EXPECTED_RESPONSE]))

        def visitor2(factory: ChainFactory):
            factory.use_prompt_template("Test {input}")

        def visitor3(factory: ChainFactory):
            factory.use_detailed_logging(True)

        factory = ChainFactory().use(visitor1).use(visitor2).use(visitor3)

        chain = factory.build()
        response = chain.invoke(TEST_INPUT)
        assert response == TEST_EXPECTED_RESPONSE


class TestPropertyGetters:
    """Test property getter methods"""

    def test_language_model_getter(self):
        """Test language_model property getter"""
        model = FakeListLLM(responses=[TEST_EXPECTED_RESPONSE])
        factory = ChainFactory().use_language_model(model)

        assert factory.language_model == model

    def test_language_model_getter_not_set(self):
        """Test language_model getter when not set"""
        factory = ChainFactory()

        with pytest.raises(RuntimeError, match="Language model is not set"):
            _ = factory.language_model

    def test_prompt_template_getter(self):
        """Test prompt_template property getter"""
        factory = ChainFactory().use_prompt_template("Test {input}")

        template = factory.prompt_template
        assert template is not None

    def test_prompt_template_getter_not_set(self):
        """Test prompt_template getter when not set"""
        factory = ChainFactory()

        with pytest.raises(RuntimeError, match="Prompt template is not set"):
            _ = factory.prompt_template

    def test_embeddings_getter(self):
        """Test embeddings property getter"""
        embeddings = FakeEmbeddings(size=3)
        factory = ChainFactory().use_embeddings(embeddings)

        assert factory.embeddings == embeddings

    def test_embeddings_getter_not_set(self):
        """Test embeddings getter when not set"""
        factory = ChainFactory()

        with pytest.raises(RuntimeError, match="Embeddings are not set"):
            _ = factory.embeddings

    def test_vector_store_getter(self):
        """Test vector_store property getter"""
        vectorstore = InMemoryVectorStore(FakeEmbeddings(size=3))
        factory = ChainFactory().use_vector_store(vectorstore)

        assert factory.vector_store == vectorstore

    def test_vector_store_getter_not_set(self):
        """Test vector_store getter when not set"""
        factory = ChainFactory()

        with pytest.raises(RuntimeError, match="Vector store is not set"):
            _ = factory.vector_store

    def test_retriever_getter(self):
        """Test retriever property getter"""

        retriever = RunnableLambda(lambda x: TEST_DOCUMENT_LIST)
        factory = ChainFactory()
        mock_vectorstore = InMemoryVectorStore(FakeEmbeddings(size=3))
        factory.use_embeddings(FakeEmbeddings(size=3))
        factory.use_vector_store(mock_vectorstore)
        factory.use_retriever(retriever)

        assert factory.retriever == retriever

    def test_text_splitter_getter(self):
        """Test text_splitter property getter"""
        splitter = RecursiveCharacterTextSplitter()
        factory = ChainFactory().use_text_splitter(splitter)

        assert factory.text_splitter == splitter

    def test_text_splitter_getter_not_set(self):
        """Test text_splitter getter when not set"""
        factory = ChainFactory()

        with pytest.raises(RuntimeError, match="Text splitter is not set"):
            _ = factory.text_splitter


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_empty_input(self):
        """Test with empty input"""
        factory = (
            ChainFactory()
            .use_prompt_template("Question: {input}\nAnswer:")
            .use_language_model(FakeListLLM(responses=[TEST_EXPECTED_RESPONSE]))
        )

        chain = factory.build()
        response = chain.invoke("")
        assert response == TEST_EXPECTED_RESPONSE

    def test_very_long_prompt_template(self):
        """Test with a very long prompt template"""
        long_template = "Context: " + ("x" * 1000) + "\nQuestion: {input}\nAnswer:"

        factory = (
            ChainFactory()
            .use_prompt_template(long_template)
            .use_language_model(FakeListLLM(responses=[TEST_EXPECTED_RESPONSE]))
        )

        chain = factory.build()
        response = chain.invoke(TEST_INPUT)
        assert response == TEST_EXPECTED_RESPONSE

    def test_special_characters_in_input(self):
        """Test with special characters in input"""
        special_input = "Test with symbols: !@#$%^&*()_+-=[]{}|;':\",./<>?"

        factory = (
            ChainFactory()
            .use_prompt_template("Process: {input}")
            .use_language_model(FakeListLLM(responses=[TEST_EXPECTED_RESPONSE]))
        )

        chain = factory.build()
        response = chain.invoke(special_input)
        assert response == TEST_EXPECTED_RESPONSE

    def test_unicode_input(self):
        """Test with unicode characters"""
        unicode_input = "Test with émojis: 你好 🎉 мир"

        factory = (
            ChainFactory()
            .use_prompt_template("Text: {input}")
            .use_language_model(FakeListLLM(responses=[TEST_EXPECTED_RESPONSE]))
        )

        chain = factory.build()
        response = chain.invoke(unicode_input)
        assert response == TEST_EXPECTED_RESPONSE
