"""Tests for the ChainFactory class"""

import pytest
from langchain_core.language_models import (
    FakeListChatModel,
    FakeListLLM,
    FakeStreamingListLLM,
    LanguageModelLike,
    LanguageModelOutput,
)
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel, Field

from quick_llm.chain_factory import ChainFactory
from quick_llm.type_definitions import ChainInputType

TEST_INPUT = "Test input"
TEST_EXPECTED_RESPONSE = "This is a sample response."
TEST_INPUT_SAMPLES: list[ChainInputType] = [
    {"input": TEST_INPUT},
    TEST_INPUT,
]
BADLY_ESCAPED_STRING = r"This is a bad \_ json string"
EXPECTED_FIXED_STRING = "This is a bad _ json string"


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


def __get_test_models(expected_response: str) -> list[LanguageModelLike]:
    return [
        FakeListLLM(responses=[expected_response]),
        FakeListChatModel(responses=[expected_response]),
    ]


def __get_test_streaming_models(expected_response: str) -> list[LanguageModelLike]:
    return [
        FakeStreamingListLLM(responses=[expected_response]),
        FakeListChatModel(responses=[expected_response]),
    ]


@pytest.mark.parametrize("input_value", TEST_INPUT_SAMPLES)
def test_string_chain_factory(input_value: ChainInputType):
    """Test the factory with a string output"""
    models = __get_test_models(TEST_EXPECTED_RESPONSE)
    for model in models:
        factory = (
            ChainFactory()
            .use_prompt_template("Sample Prompt {input}")
            .use_language_model(model)
            .use_detailed_logging()
        )
        chain = factory.build()
        response = chain.invoke(input_value)
        assert response == TEST_EXPECTED_RESPONSE


@pytest.mark.parametrize("input_value", TEST_INPUT_SAMPLES)
def test_json_chain_factory(input_value: ChainInputType):
    """Test the factory with a json output"""
    mocked_response = """{"what": "something", "when":"tomorrow", "who": "someone", "general": "something else"}"""
    models = __get_test_models(mocked_response)
    for model in models:
        factory = (
            ChainFactory.for_json_model(AnswerOutput)
            .use_prompt_template("Sample Prompt {input}")
            .use_language_model(model)
            .use_detailed_logging()
        )
        chain = factory.build()
        response = chain.invoke(input_value)
        # Checks that the response is a populated dictionary
        assert isinstance(response, dict)
        assert response.get("what", "nothing") == "something"


@pytest.mark.parametrize("input_value", TEST_INPUT_SAMPLES)
def test_stream_chain_factory(input_value: ChainInputType):
    """Test the factory with a simple text chain using streaming on a chat model and a non-chat model"""
    models = __get_test_streaming_models(TEST_EXPECTED_RESPONSE)
    for model in models:
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
def test_stream_json_chain_factory(input_value: ChainInputType):
    """Test the factory with a json output"""
    mocked_response = """{"what": "something", "when":"tomorrow", "who": "someone", "general": "something else"}"""
    models = __get_test_streaming_models(mocked_response)
    for model in models:
        factory = (
            ChainFactory.for_json_model(AnswerOutput)
            .use_prompt_template("Sample Prompt {input}")
            .use_language_model(model)
        )
        chain = factory.build()
        stream = chain.stream(input_value)
        response = list(stream)
        assert len(response) > 0
        # Not a very practical output.
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


@pytest.mark.asyncio
@pytest.mark.parametrize("input_value", TEST_INPUT_SAMPLES)
async def test_async_chain_factory(input_value: ChainInputType):
    """Test the factory with a simple text chain"""
    models = __get_test_models(TEST_EXPECTED_RESPONSE)
    for model in models:
        factory = (
            ChainFactory()
            .use_prompt_template("Sample Prompt {input}")
            .use_language_model(model)
        )
        chain = factory.build()
        response = await chain.ainvoke(input_value)
        assert response == TEST_EXPECTED_RESPONSE


@pytest.mark.asyncio
@pytest.mark.parametrize("input_value", TEST_INPUT_SAMPLES)
async def test_async_json_chain_factory(input_value: ChainInputType):
    """Test the factory with a json output"""
    mocked_response = """{"what": "something", "when":"tomorrow", "who": "someone", "general": "something else"}"""
    models = __get_test_models(mocked_response)
    for model in models:
        factory = (
            ChainFactory.for_json_model(AnswerOutput)
            .use_prompt_template("Sample Prompt {input}")
            .use_language_model(model)
        )
        chain = factory.build()
        response = await chain.ainvoke(input_value)
        # Checks that the response is a populated dictionary
        assert isinstance(response, dict)
        assert response.get("what", "nothing") == "something"


@pytest.mark.parametrize("input_value", TEST_INPUT_SAMPLES)
def test_input_transformer(input_value: ChainInputType):
    """Tests the input parser to ensure it accepts various input formats"""
    factory = ChainFactory()
    transformer = factory.input_transformer
    output = transformer.invoke(input_value)
    assert output is not None
    assert output.get(factory.input_param, "empty") == TEST_INPUT


def test_additional_values_injector():
    """Test that additional values are correctly injected for the JSON model.

    Verifies that the ChainFactory's additional_values_injector adds the necessary
    fields to the input, preserves the original input, and injects specific fields
    required for a JSON model.
    """
    factory = ChainFactory.for_json_model(AnswerOutput)
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
        (BADLY_ESCAPED_STRING, EXPECTED_FIXED_STRING),
        (
            BaseMessage(content=[BADLY_ESCAPED_STRING], type="test"),
            BaseMessage(content=[EXPECTED_FIXED_STRING], type="test"),
        ),
        (
            BaseMessage(content=BADLY_ESCAPED_STRING, type="test"),
            BaseMessage(content=EXPECTED_FIXED_STRING, type="test"),
        ),
        (
            AIMessage(content=[BADLY_ESCAPED_STRING]),
            AIMessage(content=[EXPECTED_FIXED_STRING]),
        ),
        (
            AIMessage(content=BADLY_ESCAPED_STRING),
            AIMessage(content=EXPECTED_FIXED_STRING),
        ),
    ],
)
def test_clean_output(
    test_input: LanguageModelOutput, expected_output: LanguageModelOutput
):
    """Test the text cleaning function"""
    llm = ChainFactory()
    cleaner = llm.output_cleaner
    output = cleaner.invoke(test_input)
    assert output == expected_output
