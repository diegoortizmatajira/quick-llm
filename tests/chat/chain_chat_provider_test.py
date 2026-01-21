"""Tests for the ChainChatProvider class"""

import pytest
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from quick_llm.chat import ChainChatProvider, ChatInputType
from quick_llm.chain_factory import ChainFactory

TEST_INPUT_1 = "Hello, how are you?"
TEST_INPUT_2 = "Are you ok?"
MOCKED_RESPONSE = "Mocked response"
MOCKED_CUSTOM_RESPONSE = [1, 2, 3, 4, 5]


def __build_chain_factory(responses: list | None = None) -> ChainFactory:
    """
    Create a test chain factory with mocked LLM responses.

    Args:
        responses: Optional list of responses for the fake LLM. Defaults to [MOCKED_RESPONSE].

    Returns:
        ChainFactory: Configured factory with fake chat model.
    """
    model = FakeListChatModel(responses=responses or [MOCKED_RESPONSE])
    factory = (
        ChainFactory()
        .use_prompt_template("Sample Prompt {input}")
        .use_language_model(model)
    )
    return factory


@pytest.mark.parametrize(
    ["test_input", "expected_output"],
    [
        (
            HumanMessage(content=TEST_INPUT_1),
            [TEST_INPUT_1],
        ),
        (
            [
                HumanMessage(content=[TEST_INPUT_1, TEST_INPUT_2]),
            ],
            [TEST_INPUT_1, TEST_INPUT_2],
        ),
        (
            [
                HumanMessage(content=TEST_INPUT_1),
                HumanMessage(content=TEST_INPUT_2),
            ],
            [TEST_INPUT_1, TEST_INPUT_2],
        ),
    ],
)
def test_default_input_transformer(
    test_input: ChatInputType, expected_output: list[str]
):
    """
    Test the default input transformer with various message formats.

    This test verifies that the default input transformer correctly converts
    different chat input types (single messages, lists of messages, messages
    with multiple content items) into a string format suitable for the chain.

    Args:
        test_input: Chat input in various formats (HumanMessage or list of messages).
        expected_output: List of strings that should be present in the transformed output.
    """
    llm_factory = __build_chain_factory()
    provider = ChainChatProvider(llm_factory.build)
    result = provider.default_input_transformer(test_input)
    assert result is not None
    assert isinstance(result, str)
    for expected in expected_output:
        assert expected in result


@pytest.mark.parametrize(
    ["llm_output", "expected_output"],
    [
        (MOCKED_RESPONSE, MOCKED_RESPONSE),
        ([MOCKED_RESPONSE], [MOCKED_RESPONSE]),
        ([MOCKED_RESPONSE, MOCKED_RESPONSE], [MOCKED_RESPONSE, MOCKED_RESPONSE]),
    ],
)
def test_default_output_transformer(
    llm_output: ChatInputType, expected_output: str | list[str]
):
    """
    Test the default output transformer with various output formats.

    This test verifies that the default output transformer correctly converts
    LLM outputs (strings or lists) into BaseMessage objects with appropriate
    content.

    Args:
        llm_output: LLM output in various formats (string or list).
        expected_output: Expected content of the resulting message.
    """
    llm_factory = __build_chain_factory()
    provider = ChainChatProvider(llm_factory.build)
    result = provider.default_output_transformer(llm_output)
    assert result is not None
    assert isinstance(result, BaseMessage)
    assert result.content == expected_output


def test_send():
    """
    Test synchronous message sending through the chat provider.

    This test verifies that the ChainChatProvider can send a message
    synchronously and receive a BaseMessage response with the expected content.
    """
    llm_factory = __build_chain_factory()
    provider = ChainChatProvider(llm_factory.build)
    response = provider.send(HumanMessage(content=TEST_INPUT_1))
    assert isinstance(response, BaseMessage)
    assert response.content == MOCKED_RESPONSE


async def test_send_async():
    """
    Test asynchronous message sending through the chat provider.

    This test verifies that the ChainChatProvider can send a message
    asynchronously using async/await and receive a BaseMessage response
    with the expected content.
    """
    llm_factory = __build_chain_factory()
    provider = ChainChatProvider(llm_factory.build)
    response = await provider.send_async(HumanMessage(content=TEST_INPUT_1))
    assert isinstance(response, BaseMessage)
    assert response.content == MOCKED_RESPONSE


def test_send_stream():
    """
    Test streaming message sending through the chat provider.

    This test verifies that the ChainChatProvider can send a message and
    receive a stream of BaseMessage chunks. It ensures that:
    - Each chunk is a BaseMessage
    - Multiple chunks are received
    - The chunks combine to form the complete expected response
    """
    llm_factory = __build_chain_factory()
    provider = ChainChatProvider(llm_factory.build)
    stream = provider.send_stream(HumanMessage(content=TEST_INPUT_1))
    full_output: list[str] = []
    for chunk in stream:
        assert isinstance(chunk, BaseMessage)
        assert chunk.content != MOCKED_RESPONSE
        assert isinstance(chunk.content, str)
        full_output.append(chunk.content)

    assert len(full_output) > 1
    assert "".join(full_output) == MOCKED_RESPONSE


def test_send_custom_response():
    """
    Test sending messages with a custom output transformer.

    This test verifies that ChainChatProvider can use a custom output
    transformer to convert LLM responses into custom message types with
    additional fields beyond the standard BaseMessage attributes.

    The test creates a custom message type with an extra 'custom' field
    and verifies that both the standard content and custom fields are
    correctly populated in the response.
    """
    class CustomChatMessage(AIMessage):
        """Custom chat message with additional fields."""

        custom: list[int]

    def custom_output_transformer(llm_response: dict) -> CustomChatMessage:
        """Transform dict response into CustomChatMessage."""
        return CustomChatMessage(
            content=llm_response["answer"], custom=llm_response["custom"]
        )

    llm_factory = __build_chain_factory()
    # Create a chain that returns a dict with 'answer' and 'custom' keys
    chain = llm_factory.build() | RunnableLambda(
        lambda x: {"answer": MOCKED_RESPONSE, "custom": MOCKED_CUSTOM_RESPONSE}
    )
    provider = ChainChatProvider(chain, output_transformer=custom_output_transformer)
    response = provider.send(HumanMessage(content=TEST_INPUT_1))
    assert isinstance(response, CustomChatMessage)
    assert response.content == MOCKED_RESPONSE
    assert response.custom == MOCKED_CUSTOM_RESPONSE
