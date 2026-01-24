from langchain_core.language_models import (
    FakeListChatModel,
    FakeListLLM,
    LanguageModelLike,
)
import pytest

from quick_llm.strategies import TextStrategy


TEST_INPUT = "Test input"
TEST_EXPECTED_RESPONSE = "This is a sample response."


@pytest.mark.parametrize(
    "llm",
    [
        FakeListChatModel(responses=[TEST_EXPECTED_RESPONSE]),
        FakeListLLM(responses=[TEST_EXPECTED_RESPONSE]),
    ],
)
def test_text_strategy(llm: LanguageModelLike):
    """
    Test the TextStrategy with various language models.

    Args:
        llm (LanguageModelLike): The language model to test the strategy with.

    Asserts:
        - The response is a string.
        - The response matches the expected response.
    """
    strategy = TextStrategy(llm)
    runner = strategy.adapt_llm()
    response = runner.invoke(TEST_INPUT)
    assert isinstance(response, str)
    assert response == TEST_EXPECTED_RESPONSE
