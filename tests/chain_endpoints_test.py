"""Tests for the ChainEndpoints class."""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import HumanMessage

from quick_llm import (
    ChainEndpoints,
    ChainFactory,
    ChatRequest,
    ChatResponse,
    GenerateRequest,
    GenerateResponse,
)

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
