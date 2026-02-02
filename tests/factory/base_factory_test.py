from typing import override

import pytest
from langchain_core.documents import Document
from langchain_core.language_models import FakeListLLM
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue, StringPromptValue
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.retrievers import RetrieverLike
from langchain_core.runnables.base import Runnable, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter

from quick_llm.factory.base_factory import BaseFactory
from quick_llm.support import ChainInputType, ChainOutputVar

from ..test_models import AnswerOutput

TEST_INPUT = "Test input"
TEST_INPUT_SAMPLES: list[ChainInputType] = [
    {"input": TEST_INPUT},
    TEST_INPUT,
]
TEST_EXPECTED_RESPONSE = "This is a sample response."
TEST_FAKE_DOCUMENT1 = Document(
    page_content="This is a fake document",
    metadata={"source": "FakeDocument"},
)
TEST_FAKE_DOCUMENT2 = Document(
    page_content="This is another fake document with no metadata",
)
TEST_DOCUMENT_LIST = [TEST_FAKE_DOCUMENT1, TEST_FAKE_DOCUMENT2]
TEST_EXPECTED_CUSTOM_RETRIEVER_RESULT = [TEST_FAKE_DOCUMENT2]


class TestFactory(BaseFactory):
    """A test factory for unit testing purposes."""

    def __init__(self):
        super().__init__(str)

    @override
    def build(self) -> Runnable[ChainInputType, ChainOutputVar]:
        raise NotImplementedError("This is a test factory and cannot build chains.")


class TestBaseSupportComponents:
    def test_string_prompt(self):
        """Tests the prompt template component of the ChainFactory"""
        factory = TestFactory().use_prompt_template("Sample Prompt {input}")
        prompt = factory.prompt_template
        assert prompt is not None
        rendered = prompt.invoke(
            {
                "input": TEST_INPUT,
            }
        )
        assert isinstance(rendered, StringPromptValue)
        assert rendered.text == f"Sample Prompt {TEST_INPUT}"

    def test_custom_prompt(self):
        """Tests the prompt template component of the ChainFactory"""
        factory = TestFactory().use_prompt_template(
            ChatPromptTemplate(
                [
                    ("system", "You are a helpful assistant."),
                    ("human", "Sample Prompt {input}"),
                ],
                template_format="f-string",
            )
        )
        prompt = factory.prompt_template
        assert prompt is not None
        assert isinstance(prompt, ChatPromptTemplate)
        rendered = prompt.invoke(
            {
                "input": TEST_INPUT,
            }
        )
        assert isinstance(rendered, ChatPromptValue)
        assert len(rendered.messages) == 2
        assert isinstance(rendered.messages[0], SystemMessage)
        assert rendered.messages[0].content == "You are a helpful assistant."
        assert isinstance(rendered.messages[1], HumanMessage)
        assert rendered.messages[1].content == f"Sample Prompt {TEST_INPUT}"

    def test_use_retriever(self):
        """Test the factory with a string output"""
        mock_retriever: RetrieverLike = RunnableLambda(
            lambda _: TEST_EXPECTED_CUSTOM_RETRIEVER_RESULT
        )
        factory = (
            TestFactory()
            # .use(_rag_setup_visitor(model))
            .use_retriever(mock_retriever)
        )
        retrieve_result = factory.retriever.invoke("test query")
        assert retrieve_result == TEST_EXPECTED_CUSTOM_RETRIEVER_RESULT


class TestTextSplitterConfiguration:
    """Test text splitter configuration options"""

    def test_use_text_splitter_custom(self):
        """Test using a custom text splitter"""
        custom_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100, chunk_overlap=20, separators=["\n\n", "\n", " "]
        )

        factory = TestFactory().use_text_splitter(custom_splitter)

        assert factory.text_splitter == custom_splitter

    @pytest.mark.skip(
        reason="TokenTextSplitter requires tiktoken which may not be installed"
    )
    def test_use_default_token_splitter(self):
        """Test using the default token splitter"""
        factory = TestFactory().use_default_token_splitter()

        assert factory.text_splitter is not None
        assert isinstance(factory.text_splitter, TokenTextSplitter)

    def test_use_default_text_splitter(self):
        """Test using the default text splitter"""
        factory = TestFactory().use_default_text_splitter()

        assert factory.text_splitter is not None
        assert isinstance(factory.text_splitter, RecursiveCharacterTextSplitter)


class TestGetReadableValue:
    """Test get_readable_value static method"""

    def test_get_readable_value_with_base_message(self):
        """Test readable value conversion for BaseMessage"""

        msg = HumanMessage(content="Test message")
        result = TestFactory().get_readable_value(msg)

        assert isinstance(result, str)
        assert "Test message" in result

    def test_get_readable_value_with_base_model(self):
        """Test readable value conversion for BaseModel"""
        model = AnswerOutput(
            what="test answer", when="now", who="tester", general="general info"
        )
        result = TestFactory.get_readable_value(model)

        assert isinstance(result, str)
        assert "test answer" in result

    def test_get_readable_value_with_string(self):
        """Test readable value conversion for plain string"""
        result = TestFactory.get_readable_value("plain text")
        assert result == "plain text"

    def test_get_readable_value_with_dict(self):
        """Test readable value conversion for dict"""
        test_dict = {"key": "value"}
        result = TestFactory.get_readable_value(test_dict)
        assert result == test_dict


class TestVisitorPattern:
    """Test the visitor pattern functionality"""

    def test_use_method_with_visitor(self):
        """Test using a visitor to configure the factory"""
        visited = False

        def visitor(factory: TestFactory):
            nonlocal visited
            visited = True
            factory.use_language_model(FakeListLLM(responses=[TEST_EXPECTED_RESPONSE]))
            factory.use_prompt_template("Test {input}")

        factory = TestFactory().use(visitor)

        assert visited
        assert factory.language_model is not None
        assert factory.prompt_template is not None

    def test_multiple_visitors(self):
        """Test chaining multiple visitors"""

        def visitor1(factory: TestFactory):
            factory.use_language_model(FakeListLLM(responses=[TEST_EXPECTED_RESPONSE]))

        def visitor2(factory: TestFactory):
            factory.use_prompt_template("Test {input}")

        def visitor3(factory: TestFactory):
            factory.use_detailed_logging(True)

        factory = TestFactory().use(visitor1).use(visitor2).use(visitor3)
        assert factory.language_model is not None
        assert factory.prompt_template is not None
        assert factory._detailed_logging is True
