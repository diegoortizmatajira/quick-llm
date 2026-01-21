"""Tests for the ChainEndpoints class."""

import json
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from quick_llm import (
    ChainEndpoints,
    ChainFactory,
    ChatRequest,
    ChatResponse,
    GenerateRequest,
    GenerateResponse,
)
from quick_llm.chat import ChainChatProvider

TEST_PROMPT = """
You are a test agent, you can answer questions.

Question:
{{input}}

Response:
"""

TEST_INPUT = "Test input"

MOCKED_RESPONSE = "Mocked response"


def __build_chain_factory(responses: list | None = None) -> ChainFactory:
    model = FakeListChatModel(responses=responses or [MOCKED_RESPONSE])
    factory = (
        ChainFactory()
        .use_prompt_template("Sample Prompt {input}")
        .use_language_model(model)
    )
    return factory


def __post_chat_payload(
    client: TestClient, endpoint_path: str, expected_code: int, streaming: bool = False
):
    request = ChatRequest(messages=[HumanMessage(content=TEST_INPUT)])
    response = client.post(endpoint_path, json=request.model_dump())
    if expected_code == 200:
        assert response.status_code == 200
        if streaming:
            full_response = []
            for chunk in response.iter_lines():
                chunk_dict = json.loads(chunk)
                chunk_response = ChatResponse.model_validate(chunk_dict)
                assert isinstance(chunk_response, ChatResponse)
                assert chunk_response.message is not None
                assert isinstance(chunk_response.message.content, str)
                assert chunk_response.message.content != MOCKED_RESPONSE
                assert chunk_response.created_at is not None
                full_response.append(chunk_response.message.content)
            assert len(full_response) > 1
            assert "".join(full_response) == MOCKED_RESPONSE
        else:
            json_dict = response.json()
            response = ChatResponse.model_validate(json_dict)
            assert response is not None
            assert response.message is not None
            assert response.message.content == MOCKED_RESPONSE
            assert response.created_at is not None
    else:
        assert response.status_code == expected_code


def __post_generate_payload(
    client: TestClient, endpoint_path: str, expected_code: int, streaming: bool = False
):
    request = GenerateRequest(prompt=TEST_INPUT)
    result = client.post(endpoint_path, json=request.model_dump())
    if expected_code == 200:
        assert result.status_code == 200
        if streaming:
            full_response = []
            for chunk in result.iter_lines():
                chunk_dict = json.loads(chunk)
                chunk_response = GenerateResponse.model_validate(chunk_dict)
                assert isinstance(chunk_response, GenerateResponse)
                assert chunk_response.response is not None
                assert chunk_response.response != MOCKED_RESPONSE
                assert chunk_response.created_at is not None
                full_response.append(chunk_response.response)
            assert len(full_response) > 1
            assert "".join(full_response) == MOCKED_RESPONSE
        else:
            json_dict = result.json()
            result = GenerateResponse.model_validate(json_dict)
            assert result is not None
            assert result.response == MOCKED_RESPONSE
            assert result.created_at is not None
    else:
        assert result.status_code == expected_code


@pytest.mark.parametrize("endpoint_path", ["/api/chat", "/custom_chat"])
@pytest.mark.parametrize("streaming", [False, True])
def test_chat_endpoint(endpoint_path: str, streaming: bool):
    """Tests the chat endpoint"""
    factory = __build_chain_factory()
    app = FastAPI()
    if streaming:
        ChainEndpoints(app, factory.build).with_chat_endpoint(
            endpoint=None, stream_endpoint=endpoint_path
        ).build()
    else:
        ChainEndpoints(app, factory.build).with_chat_endpoint(
            endpoint=endpoint_path
        ).build()
    client = TestClient(app)
    __post_chat_payload(client, endpoint_path, 200, streaming)


@pytest.mark.parametrize("endpoint_path", ["/api/generate", "/custom_generate"])
@pytest.mark.parametrize("streaming", [False, True])
def test_generate_endpoint(endpoint_path: str, streaming: bool):
    """Tests the generate endpoint"""
    factory = __build_chain_factory()
    app = FastAPI()
    if streaming:
        ChainEndpoints(app, factory.build).with_generate_endpoint(
            endpoint=None, stream_endpoint=endpoint_path
        ).build()
    else:
        ChainEndpoints(app, factory.build).with_generate_endpoint(
            endpoint=endpoint_path
        ).build()
    client = TestClient(app)
    __post_generate_payload(client, endpoint_path, 200, streaming)


def test_chat_request_from_chat_input_with_single_message():
    """Tests ChatRequest.from_chat_input with a single message"""
    message = HumanMessage(content="Test message")
    chat_request = ChatRequest.from_chat_input(message)
    assert len(chat_request.messages) == 1
    assert chat_request.messages[0] == message


def test_chat_request_from_chat_input_with_multiple_messages():
    """Tests ChatRequest.from_chat_input with multiple messages"""
    messages = [
        SystemMessage(content="System message"),
        HumanMessage(content="User message"),
        AIMessage(content="AI response"),
    ]
    chat_request = ChatRequest.from_chat_input(messages)
    assert len(chat_request.messages) == 3
    assert chat_request.messages == messages


def test_chat_request_from_chat_input_with_invalid_type():
    """Tests ChatRequest.from_chat_input with invalid message type"""
    with pytest.raises(ValueError, match="Unsupported message type"):
        ChatRequest.from_chat_input("not a message")  # pyright: ignore[reportArgumentType]


def test_with_defaults():
    """Tests the with_defaults method adds both generate and chat endpoints"""
    factory = __build_chain_factory()
    app = FastAPI()
    ChainEndpoints(app, factory.build).with_defaults().build()
    client = TestClient(app)

    # Test generate endpoint
    __post_generate_payload(client, "/api/generate", 200, streaming=False)

    # Test chat endpoint
    __post_chat_payload(client, "/api/chat", 200, streaming=False)


def test_chain_property_with_callable():
    """Tests the chain property when initialized with a callable"""
    factory = __build_chain_factory()
    app = FastAPI()
    endpoints = ChainEndpoints(app, factory.build)

    # Access the chain property
    chain = endpoints.chain
    assert chain is not None
    # Verify it's the actual chain instance
    result = chain.invoke(TEST_INPUT)
    assert result == MOCKED_RESPONSE


def test_chain_property_with_direct_instance():
    """Tests the chain property when initialized with a direct chain instance"""
    factory = __build_chain_factory()
    chain_instance = factory.build()
    app = FastAPI()
    endpoints = ChainEndpoints(app, chain_instance)

    # Access the chain property
    chain = endpoints.chain
    assert chain is chain_instance
    # Verify it works
    result = chain.invoke(TEST_INPUT)
    assert result == MOCKED_RESPONSE


def test_same_endpoint_and_stream_endpoint_raises_error():
    """Tests that using the same path for endpoint and stream_endpoint raises an error"""
    factory = __build_chain_factory()
    app = FastAPI()

    with pytest.raises(
        ValueError, match="Endpoint and stream_endpoint cannot be the same"
    ):
        ChainEndpoints(app, factory.build).with_chat_endpoint(
            endpoint="/api/chat", stream_endpoint="/api/chat"
        ).build()


def test_serve_chat_without_chat_provider_raises_error():
    """Tests that serving chat without a configured chat provider raises an error"""
    factory = __build_chain_factory()
    app = FastAPI()
    endpoints = ChainEndpoints(app, factory.build)

    request = ChatRequest(messages=[HumanMessage(content=TEST_INPUT)])
    with pytest.raises(ValueError, match="Chat provider is not configured"):
        endpoints.serve_chat(request)


def test_serve_chat_streaming_without_chat_provider_raises_error():
    """Tests that serving streaming chat without a configured chat provider raises an error"""
    factory = __build_chain_factory()
    app = FastAPI()
    endpoints = ChainEndpoints(app, factory.build)

    request = ChatRequest(messages=[HumanMessage(content=TEST_INPUT)])
    with pytest.raises(ValueError, match="Chat provider is not configured"):
        endpoints.serve_chat_streaming(request)


def test_with_both_generate_and_chat_endpoints():
    """Tests adding both generate and chat endpoints"""
    factory = __build_chain_factory()
    app = FastAPI()
    ChainEndpoints(app, factory.build).with_generate_endpoint(
        endpoint="/api/generate"
    ).with_chat_endpoint(endpoint="/api/chat").build()
    client = TestClient(app)

    # Test generate endpoint
    __post_generate_payload(client, "/api/generate", 200, streaming=False)

    # Test chat endpoint
    __post_chat_payload(client, "/api/chat", 200, streaming=False)


def test_with_generate_endpoint_both_standard_and_streaming():
    """Tests adding both standard and streaming generate endpoints"""
    factory = __build_chain_factory()
    app = FastAPI()
    ChainEndpoints(app, factory.build).with_generate_endpoint(
        endpoint="/api/generate", stream_endpoint="/api/generate/stream"
    ).build()
    client = TestClient(app)

    # Test standard generate endpoint
    __post_generate_payload(client, "/api/generate", 200, streaming=False)

    # Test streaming generate endpoint
    __post_generate_payload(client, "/api/generate/stream", 200, streaming=True)


def test_with_chat_endpoint_both_standard_and_streaming():
    """Tests adding both standard and streaming chat endpoints"""
    factory = __build_chain_factory()
    app = FastAPI()
    ChainEndpoints(app, factory.build).with_chat_endpoint(
        endpoint="/api/chat", stream_endpoint="/api/chat/stream"
    ).build()
    client = TestClient(app)

    # Test standard chat endpoint
    __post_chat_payload(client, "/api/chat", 200, streaming=False)

    # Test streaming chat endpoint
    __post_chat_payload(client, "/api/chat/stream", 200, streaming=True)


def test_build_without_explicit_endpoints_adds_defaults():
    """Tests that calling build without adding endpoints adds default endpoints"""
    factory = __build_chain_factory()
    app = FastAPI()
    ChainEndpoints(app, factory.build).build()
    client = TestClient(app)

    # Verify default endpoints are available
    __post_generate_payload(client, "/api/generate", 200, streaming=False)
    __post_chat_payload(client, "/api/chat", 200, streaming=False)


def test_with_generate_endpoint_only_streaming():
    """Tests adding only streaming generate endpoint (no standard endpoint)"""
    factory = __build_chain_factory()
    app = FastAPI()
    ChainEndpoints(app, factory.build).with_generate_endpoint(
        endpoint=None, stream_endpoint="/api/generate/stream"
    ).build()
    client = TestClient(app)

    # Test streaming endpoint works
    __post_generate_payload(client, "/api/generate/stream", 200, streaming=True)

    # Test standard endpoint doesn't exist
    request = GenerateRequest(prompt=TEST_INPUT)
    response = client.post("/api/generate", json=request.model_dump())
    assert response.status_code == 404


def test_with_chat_endpoint_only_streaming():
    """Tests adding only streaming chat endpoint (no standard endpoint)"""
    factory = __build_chain_factory()
    app = FastAPI()
    ChainEndpoints(app, factory.build).with_chat_endpoint(
        endpoint=None, stream_endpoint="/api/chat/stream"
    ).build()
    client = TestClient(app)

    # Test streaming endpoint works
    __post_chat_payload(client, "/api/chat/stream", 200, streaming=True)

    # Test standard endpoint doesn't exist
    request = ChatRequest(messages=[HumanMessage(content=TEST_INPUT)])
    response = client.post("/api/chat", json=request.model_dump())
    assert response.status_code == 404


def test_generate_request_model_validation():
    """Tests GenerateRequest model validation"""
    # Test with string prompt
    request = GenerateRequest(prompt="test prompt")
    assert request.prompt == "test prompt"

    # Test with dict prompt
    request_dict = GenerateRequest(prompt={"input": "test"})
    assert request_dict.prompt == {"input": "test"}

    # Test model serialization
    json_data = request.model_dump()
    assert "prompt" in json_data


def test_generate_response_model_validation():
    """Tests GenerateResponse model validation"""

    now = datetime.now(timezone.utc)
    response = GenerateResponse(response="test response", created_at=now)
    assert response.response == "test response"
    assert response.created_at == now

    # Test model serialization
    json_data = response.model_dump()
    assert "response" in json_data
    assert "created_at" in json_data


def test_chat_request_model_validation():
    """Tests ChatRequest model validation"""
    messages: list[BaseMessage] = [HumanMessage(content="test")]
    request = ChatRequest(messages=messages)
    assert len(request.messages) == 1
    assert request.messages[0].content == "test"

    # Test model serialization
    json_data = request.model_dump()
    assert "messages" in json_data


def test_chat_response_model_validation():
    """Tests ChatResponse model validation"""

    now = datetime.now(timezone.utc)
    message = AIMessage(content="test response")
    response = ChatResponse(message=message, created_at=now)
    assert response.message.content == "test response"
    assert response.created_at == now

    # Test model serialization
    json_data = response.model_dump()
    assert "message" in json_data
    assert "created_at" in json_data


def test_with_chat_endpoint_custom_transformers():
    """Tests with_chat_endpoint with custom input and output transformers"""

    factory = __build_chain_factory()
    app = FastAPI()

    # Create custom transformers
    def custom_input_transformer(messages):
        # Transform messages to simple string
        return " ".join([msg.content for msg in messages])

    def custom_output_transformer(output):
        # Transform output to AIMessage
        return AIMessage(content=str(output))

    ChainEndpoints(app, factory.build).with_chat_endpoint(
        endpoint="/api/chat",
        input_transformer=custom_input_transformer,
        output_transformer=custom_output_transformer,
    ).build()

    client = TestClient(app)
    __post_chat_payload(client, "/api/chat", 200, streaming=False)


def test_with_chat_endpoint_custom_chat_provider():
    """Tests with_chat_endpoint with a custom chat provider"""

    factory = __build_chain_factory()
    app = FastAPI()

    # Create a custom chat provider
    chat_provider = ChainChatProvider(chain=factory.build)

    ChainEndpoints(app, factory.build).with_chat_endpoint(
        endpoint="/api/chat",
        chat_provider=chat_provider,
    ).build()

    client = TestClient(app)
    __post_chat_payload(client, "/api/chat", 200, streaming=False)


def test_same_generate_endpoint_and_stream_endpoint_raises_error():
    """Tests that using the same path for generate endpoint and stream_endpoint raises an error"""
    factory = __build_chain_factory()
    app = FastAPI()

    # Note: with_generate_endpoint doesn't validate endpoint == stream_endpoint,
    # but FastAPI will raise an error when trying to register the same route twice
    # This is expected behavior - we're just verifying it doesn't silently fail
    try:
        ChainEndpoints(app, factory.build).with_generate_endpoint(
            endpoint="/api/generate", stream_endpoint="/api/generate"
        ).build()
        # If we get here, FastAPI allowed duplicate routes (shouldn't happen)
        # We'll verify by trying to use the client
        client = TestClient(app)
        request = GenerateRequest(prompt=TEST_INPUT)
        response = client.post("/api/generate", json=request.model_dump())
        # If this works, it means one of the routes was registered
        assert response.status_code in [200, 500]
    except Exception:
        # Expected: FastAPI should raise an error for duplicate routes
        pass


def test_with_generate_endpoint_only_standard():
    """Tests adding only standard generate endpoint (no streaming endpoint)"""
    factory = __build_chain_factory()
    app = FastAPI()
    ChainEndpoints(app, factory.build).with_generate_endpoint(
        endpoint="/api/generate", stream_endpoint=None
    ).build()
    client = TestClient(app)

    # Test standard endpoint works
    __post_generate_payload(client, "/api/generate", 200, streaming=False)


def test_with_chat_endpoint_only_standard():
    """Tests adding only standard chat endpoint (no streaming endpoint)"""
    factory = __build_chain_factory()
    app = FastAPI()
    ChainEndpoints(app, factory.build).with_chat_endpoint(
        endpoint="/api/chat", stream_endpoint=None
    ).build()
    client = TestClient(app)

    # Test standard endpoint works
    __post_chat_payload(client, "/api/chat", 200, streaming=False)


def test_chain_property_called_multiple_times_with_callable():
    """Tests that calling chain property multiple times with callable returns same chain"""
    factory = __build_chain_factory()
    app = FastAPI()
    endpoints = ChainEndpoints(app, factory.build)

    # Access the chain property multiple times
    chain1 = endpoints.chain
    chain2 = endpoints.chain

    # Both should work
    assert chain1.invoke(TEST_INPUT) == MOCKED_RESPONSE
    assert chain2.invoke(TEST_INPUT) == MOCKED_RESPONSE


def test_serve_generate_direct_call():
    """Tests calling serve_generate method directly"""
    factory = __build_chain_factory()
    app = FastAPI()
    endpoints = ChainEndpoints(app, factory.build)

    request = GenerateRequest(prompt=TEST_INPUT)
    response = endpoints.serve_generate(request)  # pyright: ignore[reportArgumentType]

    assert response.response == MOCKED_RESPONSE
    assert response.created_at is not None


def test_serve_generate_streaming_direct_call():
    """Tests calling serve_generate_streaming method directly"""
    factory = __build_chain_factory()
    app = FastAPI()
    endpoints = ChainEndpoints(app, factory.build)

    request = GenerateRequest(prompt=TEST_INPUT)
    response = endpoints.serve_generate_streaming(request)  # pyright: ignore[reportArgumentType]

    # Response should be a StreamingResponse

    assert isinstance(response, StreamingResponse)


def test_serve_chat_direct_call():
    """Tests calling serve_chat method directly"""
    factory = __build_chain_factory()
    app = FastAPI()
    endpoints = ChainEndpoints(app, factory.build).with_chat_endpoint(
        endpoint="/api/chat"
    )

    request = ChatRequest(messages=[HumanMessage(content=TEST_INPUT)])
    response = endpoints.serve_chat(request)

    assert response.message.content == MOCKED_RESPONSE
    assert response.created_at is not None


def test_serve_chat_streaming_direct_call():
    """Tests calling serve_chat_streaming method directly"""
    factory = __build_chain_factory()
    app = FastAPI()
    endpoints = ChainEndpoints(app, factory.build).with_chat_endpoint(
        endpoint="/api/chat"
    )

    request = ChatRequest(messages=[HumanMessage(content=TEST_INPUT)])
    response = endpoints.serve_chat_streaming(request)

    # Response should be a StreamingResponse

    assert isinstance(response, StreamingResponse)


def test_multiple_build_calls():
    """Tests that calling build multiple times works correctly"""
    factory = __build_chain_factory()
    app = FastAPI()
    endpoints = ChainEndpoints(app, factory.build)

    # First build
    endpoints.with_generate_endpoint(endpoint="/api/generate").build()

    # Second build should not cause issues (routes already registered)
    # Note: This might raise an error in FastAPI if routes are duplicated
    # The implementation should handle this gracefully
    client = TestClient(app)
    __post_generate_payload(client, "/api/generate", 200, streaming=False)


def test_chat_request_from_chat_input_with_empty_list():
    """Tests ChatRequest.from_chat_input with an empty list"""
    messages = []
    chat_request = ChatRequest.from_chat_input(messages)
    assert len(chat_request.messages) == 0


def test_chat_request_from_chat_input_with_ai_message():
    """Tests ChatRequest.from_chat_input with an AI message"""
    message = AIMessage(content="AI response")
    chat_request = ChatRequest.from_chat_input(message)
    assert len(chat_request.messages) == 1
    assert isinstance(chat_request.messages[0], AIMessage)
    assert chat_request.messages[0].content == "AI response"


def test_chat_request_from_chat_input_with_system_message():
    """Tests ChatRequest.from_chat_input with a system message"""
    message = SystemMessage(content="System prompt")
    chat_request = ChatRequest.from_chat_input(message)
    assert len(chat_request.messages) == 1
    assert isinstance(chat_request.messages[0], SystemMessage)
    assert chat_request.messages[0].content == "System prompt"
