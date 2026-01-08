"""Tests for the ChainFactory class"""

import pytest
from langchain_core.language_models import (
    BaseLanguageModel,
    FakeListChatModel,
    FakeListLLM,
)
from pydantic import BaseModel, Field

from quick_llm.chain_factory import ChainFactory
from quick_llm.type_definitions import ChainInputType

TEST_INPUT = "Test input"
TEST_EXPECTED_RESPONSE = "This is a sample response."
TEST_INPUT_SAMPLES: list[ChainInputType] = [
    {"input": TEST_INPUT},
    TEST_INPUT,
]


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


def __get_test_models(expected_response: str) -> list[BaseLanguageModel]:
    return [
        FakeListLLM(responses=[expected_response]),
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
        )
        chain = factory.build()
        response = chain.invoke(input_value)
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
