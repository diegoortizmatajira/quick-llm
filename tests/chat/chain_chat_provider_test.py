"""Tests for the ChainChatProvider class"""

import pytest
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda

from quick_llm import ChainFactory, ChainInputType
from quick_llm.chat import ChainChatProvider, ChatInputType

TEST_INPUT_1 = "Hello, how are you?"
TEST_INPUT_2 = "Are you ok?"
MOCKED_RESPONSE = "Mocked response"
MOCKED_CUSTOM_RESPONSE = [1, 2, 3, 4, 5]


def _build_chain_factory(responses: list | None = None) -> ChainFactory:
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
    llm_factory = _build_chain_factory()
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
    llm_factory = _build_chain_factory()
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
    llm_factory = _build_chain_factory()
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
    llm_factory = _build_chain_factory()
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
    llm_factory = _build_chain_factory()
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

    llm_factory = _build_chain_factory()
    # Create a chain that returns a dict with 'answer' and 'custom' keys
    chain = llm_factory.build() | RunnableLambda(
        lambda x: {"answer": MOCKED_RESPONSE, "custom": MOCKED_CUSTOM_RESPONSE}
    )
    provider = ChainChatProvider(chain, output_transformer=custom_output_transformer)
    response = provider.send(HumanMessage(content=TEST_INPUT_1))
    assert isinstance(response, CustomChatMessage)
    assert response.content == MOCKED_RESPONSE
    assert response.custom == MOCKED_CUSTOM_RESPONSE


class TestChainChatProviderInitialization:
    """Test ChainChatProvider initialization with various configurations."""

    def test_initialization_with_chain_instance(self):
        """Test initialization with a chain instance (not factory)."""
        factory = _build_chain_factory()
        chain = factory.build()
        provider = ChainChatProvider(chain)

        response = provider.send(HumanMessage(content=TEST_INPUT_1))
        assert isinstance(response, BaseMessage)
        assert response.content == MOCKED_RESPONSE

    def test_initialization_with_factory_function(self):
        """Test initialization with a factory function."""
        factory = _build_chain_factory()
        provider = ChainChatProvider(factory.build)

        response = provider.send(HumanMessage(content=TEST_INPUT_1))
        assert isinstance(response, BaseMessage)
        assert response.content == MOCKED_RESPONSE

    def test_initialization_with_custom_input_transformer(self):
        """Test initialization with custom input transformer."""

        def custom_input_transformer(msg: ChatInputType) -> ChainInputType:
            """Extract only content from message."""
            if isinstance(msg, BaseMessage):
                return f"CUSTOM: {msg.content}"
            return "CUSTOM: " + str(msg)

        factory = _build_chain_factory()
        provider = ChainChatProvider(
            factory.build, input_transformer=custom_input_transformer
        )

        response = provider.send(HumanMessage(content=TEST_INPUT_1))
        assert isinstance(response, BaseMessage)

    def test_initialization_with_both_custom_transformers(self):
        """Test initialization with both custom transformers."""

        def custom_input_transformer(msg: ChatInputType) -> ChainInputType:
            return f"INPUT: {msg}"

        def custom_output_transformer(output: str) -> BaseMessage:
            return AIMessage(content=f"OUTPUT: {output}")

        factory = _build_chain_factory()
        provider = ChainChatProvider(
            factory.build,
            input_transformer=custom_input_transformer,
            output_transformer=custom_output_transformer,
        )

        response = provider.send(HumanMessage(content=TEST_INPUT_1))
        assert isinstance(response, BaseMessage)
        assert "OUTPUT:" in response.content

    def test_factory_called_each_time(self):
        """Test that factory function is called for each request."""
        call_count = 0

        def counting_factory():
            nonlocal call_count
            call_count += 1
            return _build_chain_factory().build()

        provider = ChainChatProvider(counting_factory)

        provider.send(HumanMessage(content=TEST_INPUT_1))
        assert call_count == 1

        provider.send(HumanMessage(content=TEST_INPUT_2))
        assert call_count == 2


class TestDefaultInputTransformerEdgeCases:
    """Test default input transformer with edge cases."""

    def test_with_string_input(self):
        """Test input transformer with plain string input raises error."""
        factory = _build_chain_factory()
        provider = ChainChatProvider(factory.build)

        # Plain strings are not valid ChatInputType (must be wrapped in BaseMessage)
        with pytest.raises(AttributeError):
            provider.default_input_transformer("plain string")  # pyright: ignore[reportArgumentType]

    def test_with_single_basemessage(self):
        """Test input transformer with single BaseMessage (not in list)."""
        factory = _build_chain_factory()
        provider = ChainChatProvider(factory.build)

        msg = HumanMessage(content=TEST_INPUT_1)
        result = provider.default_input_transformer(msg)
        assert isinstance(result, str)
        assert TEST_INPUT_1 in result

    def test_with_empty_list(self):
        """Test input transformer with empty list."""
        factory = _build_chain_factory()
        provider = ChainChatProvider(factory.build)

        result = provider.default_input_transformer([])
        assert isinstance(result, str)
        assert result == ""

    def test_with_mixed_message_types(self):
        """Test input transformer with mixed message types."""
        factory = _build_chain_factory()
        provider = ChainChatProvider(factory.build)

        messages = [
            HumanMessage(content=TEST_INPUT_1),
            AIMessage(content="AI response"),
            SystemMessage(content="System message"),
        ]
        result = provider.default_input_transformer(messages)
        assert isinstance(result, str)
        assert TEST_INPUT_1 in result
        assert "AI response" in result
        assert "System message" in result

    def test_with_aimessage(self):
        """Test input transformer with AIMessage."""
        factory = _build_chain_factory()
        provider = ChainChatProvider(factory.build)

        msg = AIMessage(content="AI message")
        result = provider.default_input_transformer(msg)
        assert isinstance(result, str)
        assert "AI message" in result

    def test_with_systemmessage(self):
        """Test input transformer with SystemMessage."""
        factory = _build_chain_factory()
        provider = ChainChatProvider(factory.build)

        msg = SystemMessage(content="System instruction")
        result = provider.default_input_transformer(msg)
        assert isinstance(result, str)
        assert "System instruction" in result


class TestDefaultOutputTransformerEdgeCases:
    """Test default output transformer with edge cases."""

    def test_with_basemessage_output(self):
        """Test output transformer with BaseMessage (should return as-is)."""
        factory = _build_chain_factory()
        provider = ChainChatProvider(factory.build)

        message = AIMessage(content="AI response")
        result = provider.default_output_transformer(message)
        assert result is message
        assert isinstance(result, BaseMessage)

    def test_with_aimessage_output(self):
        """Test output transformer with AIMessage."""
        factory = _build_chain_factory()
        provider = ChainChatProvider(factory.build)

        message = AIMessage(content="Direct AI message")
        result = provider.default_output_transformer(message)
        assert result is message

    def test_with_integer_output(self):
        """Test output transformer with integer output."""
        factory = _build_chain_factory()
        provider = ChainChatProvider(factory.build)

        result = provider.default_output_transformer(42)
        assert isinstance(result, AIMessage)
        assert result.content == "42"

    def test_with_float_output(self):
        """Test output transformer with float output."""
        factory = _build_chain_factory()
        provider = ChainChatProvider(factory.build)

        result = provider.default_output_transformer(3.14)
        assert isinstance(result, AIMessage)
        assert result.content == "3.14"

    def test_with_dict_output(self):
        """Test output transformer with dict output."""
        factory = _build_chain_factory()
        provider = ChainChatProvider(factory.build)

        result = provider.default_output_transformer({"key": "value"})
        assert isinstance(result, AIMessage)
        assert "key" in result.content
        assert "value" in result.content

    def test_with_empty_string_output(self):
        """Test output transformer with empty string."""
        factory = _build_chain_factory()
        provider = ChainChatProvider(factory.build)

        result = provider.default_output_transformer("")
        assert isinstance(result, AIMessage)
        assert result.content == ""

    def test_with_empty_list_output(self):
        """Test output transformer with empty list."""
        factory = _build_chain_factory()
        provider = ChainChatProvider(factory.build)

        result = provider.default_output_transformer([])
        assert isinstance(result, AIMessage)
        assert result.content == []

    def test_with_custom_object_output(self):
        """Test output transformer with custom object."""

        class CustomObject:
            def __str__(self):
                return "Custom String Representation"

        factory = _build_chain_factory()
        provider = ChainChatProvider(factory.build)

        result = provider.default_output_transformer(CustomObject())
        assert isinstance(result, AIMessage)
        assert result.content == "Custom String Representation"


class TestCustomTransformers:
    """Test custom input and output transformers."""

    def test_custom_input_transformer_extracts_content(self):
        """Test custom input transformer that extracts content."""

        def extract_content(msg: ChatInputType) -> ChainInputType:
            if isinstance(msg, BaseMessage):
                return msg.content
            return str(msg)

        factory = _build_chain_factory()
        provider = ChainChatProvider(factory.build, input_transformer=extract_content)

        response = provider.send(HumanMessage(content=TEST_INPUT_1))
        assert isinstance(response, BaseMessage)

    def test_custom_input_transformer_returns_dict(self):
        """Test custom input transformer that returns dict."""

        def to_dict_transformer(msg: ChatInputType) -> ChainInputType:
            return {"user_input": str(msg)}

        factory = (
            ChainFactory()
            .use_prompt_template("Process: {user_input}")
            .use_language_model(FakeListChatModel(responses=[MOCKED_RESPONSE]))
        )
        provider = ChainChatProvider(
            factory.build, input_transformer=to_dict_transformer
        )

        response = provider.send(HumanMessage(content=TEST_INPUT_1))
        assert isinstance(response, BaseMessage)

    def test_custom_output_transformer_adds_metadata(self):
        """Test custom output transformer that adds metadata."""

        def add_metadata(output: str) -> BaseMessage:
            return AIMessage(content=output, additional_kwargs={"processed": True})

        factory = _build_chain_factory()
        provider = ChainChatProvider(factory.build, output_transformer=add_metadata)

        response = provider.send(HumanMessage(content=TEST_INPUT_1))
        assert response.additional_kwargs.get("processed") is True


class TestErrorHandling:
    """Test error handling in ChainChatProvider."""

    def test_chain_invoke_raises_exception(self):
        """Test handling when chain invoke raises exception."""

        def failing_chain():
            chain = RunnableLambda(lambda x: 1 / 0)  # Raises ZeroDivisionError
            return chain

        provider = ChainChatProvider(failing_chain)

        with pytest.raises(ZeroDivisionError):
            provider.send(HumanMessage(content=TEST_INPUT_1))

    async def test_chain_ainvoke_raises_exception(self):
        """Test handling when chain ainvoke raises exception."""

        def failing_chain():
            chain = RunnableLambda(lambda x: 1 / 0)
            return chain

        provider = ChainChatProvider(failing_chain)

        with pytest.raises(ZeroDivisionError):
            await provider.send_async(HumanMessage(content=TEST_INPUT_1))

    def test_chain_stream_raises_exception(self):
        """Test handling when chain stream raises exception."""

        def failing_stream_chain():
            def stream_generator(_):
                yield "first"
                raise RuntimeError("Stream failed")

            return RunnableLambda(stream_generator)

        provider = ChainChatProvider(failing_stream_chain)

        with pytest.raises(RuntimeError, match="Stream failed"):
            stream = provider.send_stream(HumanMessage(content=TEST_INPUT_1))
            list(stream)  # Consume the stream to trigger the error

    def test_input_transformer_raises_exception(self):
        """Test handling when input transformer raises exception."""

        def failing_transformer(msg: ChatInputType) -> ChainInputType:
            raise ValueError("Transformation failed")

        factory = _build_chain_factory()
        provider = ChainChatProvider(
            factory.build, input_transformer=failing_transformer
        )

        with pytest.raises(ValueError, match="Transformation failed"):
            provider.send(HumanMessage(content=TEST_INPUT_1))

    def test_output_transformer_raises_exception(self):
        """Test handling when output transformer raises exception."""

        def failing_transformer(_) -> BaseMessage:
            raise ValueError("Output transformation failed")

        factory = _build_chain_factory()
        provider = ChainChatProvider(
            factory.build, output_transformer=failing_transformer
        )

        with pytest.raises(ValueError, match="Output transformation failed"):
            provider.send(HumanMessage(content=TEST_INPUT_1))


class TestSendMethodVariations:
    """Test various send method scenarios."""

    def test_multiple_sequential_sends(self):
        """Test multiple sequential send calls."""
        responses = ["Response 1", "Response 2", "Response 3"]
        factory = _build_chain_factory(responses)
        provider = ChainChatProvider(factory.build)

        for expected_response in responses:
            response = provider.send(HumanMessage(content=TEST_INPUT_1))
            assert isinstance(response, BaseMessage)
            assert response.content == expected_response

    async def test_multiple_sequential_async_sends(self):
        """Test multiple sequential async send calls."""
        responses = ["Async 1", "Async 2", "Async 3"]
        factory = _build_chain_factory(responses)
        provider = ChainChatProvider(factory.build)

        for expected_response in responses:
            response = await provider.send_async(HumanMessage(content=TEST_INPUT_1))
            assert isinstance(response, BaseMessage)
            assert response.content == expected_response

    def test_send_with_different_message_types(self):
        """Test send with different message types."""
        factory = _build_chain_factory([MOCKED_RESPONSE] * 3)
        provider = ChainChatProvider(factory.build)

        # HumanMessage
        response1 = provider.send(HumanMessage(content=TEST_INPUT_1))
        assert isinstance(response1, BaseMessage)

        # AIMessage
        response2 = provider.send(AIMessage(content=TEST_INPUT_1))
        assert isinstance(response2, BaseMessage)

        # SystemMessage
        response3 = provider.send(SystemMessage(content=TEST_INPUT_1))
        assert isinstance(response3, BaseMessage)


class TestStreamEdgeCases:
    """Test streaming edge cases."""

    def test_stream_with_single_chunk(self):
        """Test streaming when only one chunk is returned."""
        short_response = "OK"
        factory = _build_chain_factory([short_response])
        provider = ChainChatProvider(factory.build)

        chunks = list(provider.send_stream(HumanMessage(content=TEST_INPUT_1)))

        assert len(chunks) >= 1
        assert all(isinstance(chunk, BaseMessage) for chunk in chunks)

        # Combine all chunks
        full_content = "".join(str(chunk.content) for chunk in chunks)
        assert full_content == short_response

    def test_stream_with_many_chunks(self):
        """Test streaming with many chunks."""
        long_response = "A" * 100  # Long response should stream in chunks
        factory = _build_chain_factory([long_response])
        provider = ChainChatProvider(factory.build)

        chunks = list(provider.send_stream(HumanMessage(content=TEST_INPUT_1)))

        assert len(chunks) >= 1
        assert all(isinstance(chunk, BaseMessage) for chunk in chunks)

        full_content = "".join(str(chunk.content) for chunk in chunks)
        assert full_content == long_response

    def test_stream_multiple_times(self):
        """Test calling stream multiple times."""
        responses = ["Stream 1", "Stream 2"]
        factory = _build_chain_factory(responses)
        provider = ChainChatProvider(factory.build)

        # First stream
        chunks1 = list(provider.send_stream(HumanMessage(content=TEST_INPUT_1)))
        content1 = "".join(str(c.content) for c in chunks1)
        assert content1 == "Stream 1"

        # Second stream
        chunks2 = list(provider.send_stream(HumanMessage(content=TEST_INPUT_2)))
        content2 = "".join(str(c.content) for c in chunks2)
        assert content2 == "Stream 2"


class TestIntegrationScenarios:
    """Test integration scenarios with complex chains."""

    def test_complex_chain_with_transformers(self):
        """Test complex chain with input and output transformers."""

        def uppercase_input(msg: ChatInputType) -> ChainInputType:
            if isinstance(msg, BaseMessage):
                return str(msg.content).upper()
            return str(msg).upper()

        def lowercase_output(output: str) -> BaseMessage:
            return AIMessage(content=output.lower())

        factory = _build_chain_factory(["UPPERCASE RESPONSE"])
        provider = ChainChatProvider(
            factory.build,
            input_transformer=uppercase_input,
            output_transformer=lowercase_output,
        )

        response = provider.send(HumanMessage(content="test"))
        assert response.content == "uppercase response"

    def test_chain_with_message_history(self):
        """Test chain handling message history."""
        factory = _build_chain_factory()
        provider = ChainChatProvider(factory.build)

        # Simulate conversation with multiple messages
        messages = [
            HumanMessage(content="First message"),
            AIMessage(content="First response"),
            HumanMessage(content="Second message"),
        ]

        # Send list of messages
        result = provider.default_input_transformer(messages)
        assert isinstance(result, str)
        assert "First message" in result
        assert "First response" in result
        assert "Second message" in result

    async def test_consistency_between_sync_and_async(self):
        """Test that sync and async methods produce consistent results."""
        factory = _build_chain_factory()
        chain = factory.build()
        provider = ChainChatProvider(chain)

        message = HumanMessage(content=TEST_INPUT_1)

        sync_response = provider.send(message)
        async_response = await provider.send_async(message)

        assert sync_response.content == async_response.content
        assert type(sync_response) is type(async_response)
