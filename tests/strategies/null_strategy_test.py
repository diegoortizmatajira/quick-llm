from langchain_core.language_models import (
    FakeListChatModel,
    FakeListLLM,
    LanguageModelLike,
)
from langchain_core.messages import BaseMessage
import pytest

from quick_llm import ChainFactory
from quick_llm.strategies import NullStrategy


TEST_INPUT = "Test input"
TEST_EXPECTED_RESPONSE = "This is a sample response."


@pytest.mark.parametrize(
    "llm",
    [
        FakeListChatModel(responses=[TEST_EXPECTED_RESPONSE]),
        FakeListLLM(responses=[TEST_EXPECTED_RESPONSE]),
    ],
)
def test_null_strategy(llm: LanguageModelLike):
    """
    Test the TextStrategy with various language models.

    Args:
        llm (LanguageModelLike): The language model to test the strategy with.

    Asserts:
        - The response is a string.
        - The response matches the expected response.
    """
    control_response = llm.invoke(TEST_INPUT)
    factory = ChainFactory(str).use_language_model(llm)
    strategy = NullStrategy(factory)
    runner = strategy.adapt_llm()
    response = runner.invoke(TEST_INPUT)
    if isinstance(response, BaseMessage) and isinstance(control_response, BaseMessage):
        assert response.content == control_response.content
    else:
        assert response == control_response
