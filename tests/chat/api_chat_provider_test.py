"""Tests for the APIChatProvider class"""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from quick_llm.chat.api_chat_provider import APIChatProvider
from quick_llm.chain_endpoints import ChatRequest


# Test constants
TEST_API_URL = "http://localhost:8000/api/chat"
TEST_TIMEOUT = 30
TEST_INPUT = "Hello, how are you?"
TEST_RESPONSE_CONTENT = "I'm doing well, thank you!"


def _create_mock_response(status_code: int, content: dict | str) -> Mock:
    """
    Create a mock HTTP response object.

    Args:
        status_code: HTTP status code for the response.
        content: Response content (dict for successful response, str for error messages).

    Returns:
        Mock: Mocked response object with status_code and content.
    """
    mock_response = Mock()
    mock_response.status_code = status_code

    if isinstance(content, dict):
        # For successful responses, return the dict directly
        # Note: The actual API code uses BaseMessage.model_validate(response.content)
        # which expects a dict, not bytes
        mock_response.content = content
    else:
        # Error response with plain text as bytes
        mock_response.content = content.encode("utf-8")

    return mock_response


def _create_message_json(content: str, message_type: str = "ai") -> dict:
    """
    Create a JSON representation of a BaseMessage.

    Args:
        content: The message content.
        message_type: The type of message ("ai", "human", etc.).

    Returns:
        dict: JSON-serializable dictionary representing a message.
    """
    return {
        "content": content,
        "type": message_type,
        "additional_kwargs": {},
        "response_metadata": {},
    }


class TestAPIChatProviderInitialization:
    """Test APIChatProvider initialization"""

    def test_init_with_url_only(self):
        """Test initialization with only URL parameter"""
        provider = APIChatProvider(TEST_API_URL)

        assert provider.url == TEST_API_URL
        assert provider.timeout is None
        assert provider.logger is not None

    def test_init_with_url_and_timeout(self):
        """Test initialization with URL and timeout parameters"""
        provider = APIChatProvider(TEST_API_URL, timeout=TEST_TIMEOUT)

        assert provider.url == TEST_API_URL
        assert provider.timeout == TEST_TIMEOUT
        assert provider.logger is not None

    def test_init_logs_configuration(self, caplog):
        """Test that initialization logs the URL configuration"""
        with caplog.at_level("INFO"):
            APIChatProvider(TEST_API_URL)

        assert "Configuring chat provider to use API" in caplog.text
        assert TEST_API_URL in caplog.text


class TestSendMethod:
    """Test send method with mocked requests"""

    @patch("quick_llm.chat.api_chat_provider.requests.post")
    def test_send_successful_request(self, mock_post):
        """Test successful message sending with 200 response"""
        # Setup mock response
        response_data = _create_message_json(TEST_RESPONSE_CONTENT)
        mock_post.return_value = _create_mock_response(200, response_data)

        # Create provider and send message
        provider = APIChatProvider(TEST_API_URL)
        message = [
            HumanMessage(content=TEST_INPUT)
        ]  # Wrap in list to avoid iteration issues
        result = provider.send(message)

        # Verify request was made correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args.args[0] == TEST_API_URL
        assert "json" in call_args.kwargs
        assert call_args.kwargs["timeout"] is None

        # Verify result is a BaseMessage
        assert isinstance(result, BaseMessage)

    @patch("quick_llm.chat.api_chat_provider.requests.post")
    def test_send_with_timeout(self, mock_post):
        """Test that timeout parameter is passed to requests.post"""
        response_data = _create_message_json(TEST_RESPONSE_CONTENT)
        mock_post.return_value = _create_mock_response(200, response_data)

        provider = APIChatProvider(TEST_API_URL, timeout=TEST_TIMEOUT)
        message = [HumanMessage(content=TEST_INPUT)]
        provider.send(message)

        # Verify timeout was passed
        call_args = mock_post.call_args
        assert call_args.kwargs["timeout"] == TEST_TIMEOUT

    @patch("quick_llm.chat.api_chat_provider.requests.post")
    def test_send_formats_request_correctly(self, mock_post):
        """Test that the request is formatted as ChatRequest"""
        response_data = _create_message_json(TEST_RESPONSE_CONTENT)
        mock_post.return_value = _create_mock_response(200, response_data)

        provider = APIChatProvider(TEST_API_URL)
        message = [HumanMessage(content=TEST_INPUT)]
        provider.send(message)

        # Verify request format
        call_args = mock_post.call_args
        request_data = call_args.kwargs["json"]

        assert "messages" in request_data
        assert isinstance(request_data["messages"], list)
        assert len(request_data["messages"]) > 0

    @patch("quick_llm.chat.api_chat_provider.requests.post")
    def test_send_with_multiple_messages(self, mock_post):
        """Test sending multiple messages in a list"""
        response_data = _create_message_json(TEST_RESPONSE_CONTENT)
        mock_post.return_value = _create_mock_response(200, response_data)

        provider = APIChatProvider(TEST_API_URL)
        messages = [
            HumanMessage(content="First message"),
            AIMessage(content="AI response"),
            HumanMessage(content="Second message"),
        ]
        provider.send(messages)

        # Verify request contains all messages
        call_args = mock_post.call_args
        request_data = call_args.kwargs["json"]
        assert len(request_data["messages"]) == 3

    @patch("quick_llm.chat.api_chat_provider.requests.post")
    def test_send_logs_request_content(self, mock_post, caplog):
        """Test that send logs the request content at DEBUG level"""
        response_data = _create_message_json(TEST_RESPONSE_CONTENT)
        mock_post.return_value = _create_mock_response(200, response_data)

        provider = APIChatProvider(TEST_API_URL)
        message = [HumanMessage(content=TEST_INPUT)]

        with caplog.at_level("DEBUG"):
            provider.send(message)

        assert "Sending message to API" in caplog.text

    @patch("quick_llm.chat.api_chat_provider.requests.post")
    def test_send_logs_response_content(self, mock_post, caplog):
        """Test that send logs the response content at DEBUG level"""
        response_data = _create_message_json(TEST_RESPONSE_CONTENT)
        mock_post.return_value = _create_mock_response(200, response_data)

        provider = APIChatProvider(TEST_API_URL)
        message = [HumanMessage(content=TEST_INPUT)]

        with caplog.at_level("DEBUG"):
            provider.send(message)

        assert "Response from the API" in caplog.text


class TestSendErrorHandling:
    """Test error handling in send method"""

    @patch("quick_llm.chat.api_chat_provider.requests.post")
    def test_send_with_404_error(self, mock_post):
        """Test handling of 404 Not Found error"""
        mock_post.return_value = _create_mock_response(404, "Not Found")

        provider = APIChatProvider(TEST_API_URL)
        message = [HumanMessage(content=TEST_INPUT)]
        result = provider.send(message)

        # Should return error message
        assert isinstance(result, AIMessage)
        assert "Failed to get a response from the server" in result.content

    @patch("quick_llm.chat.api_chat_provider.requests.post")
    def test_send_with_500_error(self, mock_post):
        """Test handling of 500 Internal Server Error"""
        mock_post.return_value = _create_mock_response(500, "Internal Server Error")

        provider = APIChatProvider(TEST_API_URL)
        message = [HumanMessage(content=TEST_INPUT)]
        result = provider.send(message)

        # Should return error message
        assert isinstance(result, AIMessage)
        assert "Failed to get a response from the server" in result.content

    @patch("quick_llm.chat.api_chat_provider.requests.post")
    def test_send_with_401_unauthorized(self, mock_post):
        """Test handling of 401 Unauthorized error"""
        mock_post.return_value = _create_mock_response(401, "Unauthorized")

        provider = APIChatProvider(TEST_API_URL)
        message = [HumanMessage(content=TEST_INPUT)]
        result = provider.send(message)

        # Should return error message
        assert isinstance(result, AIMessage)
        assert "Failed to get a response from the server" in result.content

    @patch("quick_llm.chat.api_chat_provider.requests.post")
    def test_send_logs_error_details(self, mock_post, caplog):
        """Test that errors are logged with status code and content"""
        mock_post.return_value = _create_mock_response(404, "Not Found")

        provider = APIChatProvider(TEST_API_URL)
        message = [HumanMessage(content=TEST_INPUT)]

        with caplog.at_level("ERROR"):
            provider.send(message)

        assert "Error sending request to API" in caplog.text
        assert "404" in caplog.text
        assert "Not Found" in caplog.text

    @patch("quick_llm.chat.api_chat_provider.requests.post")
    def test_send_with_timeout_exception(self, mock_post):
        """Test handling of request timeout exception"""
        import requests

        mock_post.side_effect = requests.Timeout("Request timed out")

        provider = APIChatProvider(TEST_API_URL, timeout=5)
        message = [HumanMessage(content=TEST_INPUT)]

        # Should raise the timeout exception
        with pytest.raises(requests.Timeout):
            provider.send(message)

    @patch("quick_llm.chat.api_chat_provider.requests.post")
    def test_send_with_connection_error(self, mock_post):
        """Test handling of connection error"""
        import requests

        mock_post.side_effect = requests.ConnectionError("Connection refused")

        provider = APIChatProvider(TEST_API_URL)
        message = [HumanMessage(content=TEST_INPUT)]

        # Should raise the connection error
        with pytest.raises(requests.ConnectionError):
            provider.send(message)


class TestSendAsyncMethod:
    """Test send_async method"""

    @patch("quick_llm.chat.api_chat_provider.requests.post")
    async def test_send_async_calls_sync_send(self, mock_post):
        """Test that send_async delegates to send method"""
        response_data = _create_message_json(TEST_RESPONSE_CONTENT)
        mock_post.return_value = _create_mock_response(200, response_data)

        provider = APIChatProvider(TEST_API_URL)
        message = [HumanMessage(content=TEST_INPUT)]
        result = await provider.send_async(message)

        # Verify request was made
        mock_post.assert_called_once()

        # Verify result is a BaseMessage
        assert isinstance(result, BaseMessage)

    @patch("quick_llm.chat.api_chat_provider.requests.post")
    async def test_send_async_with_error(self, mock_post):
        """Test that send_async handles errors like send"""
        mock_post.return_value = _create_mock_response(500, "Server Error")

        provider = APIChatProvider(TEST_API_URL)
        message = [HumanMessage(content=TEST_INPUT)]
        result = await provider.send_async(message)

        # Should return error message like send()
        assert isinstance(result, AIMessage)
        assert "Failed to get a response from the server" in result.content


class TestSendStreamMethod:
    """Test send_stream method"""

    @patch("quick_llm.chat.api_chat_provider.requests.post")
    def test_send_stream_yields_result(self, mock_post):
        """Test that send_stream yields the message result"""
        response_data = _create_message_json(TEST_RESPONSE_CONTENT)
        mock_post.return_value = _create_mock_response(200, response_data)

        provider = APIChatProvider(TEST_API_URL)
        message = [HumanMessage(content=TEST_INPUT)]
        stream = provider.send_stream(message)

        # Collect all yielded messages
        results = list(stream)

        # Should yield exactly one message
        assert len(results) == 1
        assert isinstance(results[0], BaseMessage)

    @patch("quick_llm.chat.api_chat_provider.requests.post")
    def test_send_stream_with_error(self, mock_post):
        """Test that send_stream yields error message on failure"""
        mock_post.return_value = _create_mock_response(404, "Not Found")

        provider = APIChatProvider(TEST_API_URL)
        message = [HumanMessage(content=TEST_INPUT)]
        stream = provider.send_stream(message)

        # Collect all yielded messages
        results = list(stream)

        # Should yield error message
        assert len(results) == 1
        assert isinstance(results[0], AIMessage)
        assert "Failed to get a response from the server" in results[0].content

    @patch("quick_llm.chat.api_chat_provider.requests.post")
    def test_send_stream_makes_single_request(self, mock_post):
        """Test that send_stream makes only one API request"""
        response_data = _create_message_json(TEST_RESPONSE_CONTENT)
        mock_post.return_value = _create_mock_response(200, response_data)

        provider = APIChatProvider(TEST_API_URL)
        message = [HumanMessage(content=TEST_INPUT)]
        stream = provider.send_stream(message)

        # Consume the stream
        list(stream)

        # Should have made exactly one request
        mock_post.assert_called_once()


class TestChatRequestIntegration:
    """Test integration with ChatRequest"""

    @patch("quick_llm.chat.api_chat_provider.requests.post")
    def test_send_creates_valid_chat_request(self, mock_post):
        """Test that send creates a valid ChatRequest"""
        response_data = _create_message_json(TEST_RESPONSE_CONTENT)
        mock_post.return_value = _create_mock_response(200, response_data)

        provider = APIChatProvider(TEST_API_URL)
        message = [HumanMessage(content=TEST_INPUT)]
        provider.send(message)

        # Extract the request data
        call_args = mock_post.call_args
        request_data = call_args.kwargs["json"]

        # Verify it can be reconstructed as ChatRequest
        chat_request = ChatRequest.model_validate(request_data)
        assert isinstance(chat_request, ChatRequest)
        assert len(chat_request.messages) > 0


class TestDifferentMessageTypes:
    """Test handling of different message types"""

    @patch("quick_llm.chat.api_chat_provider.requests.post")
    def test_send_with_human_message(self, mock_post):
        """Test sending a HumanMessage"""
        response_data = _create_message_json(TEST_RESPONSE_CONTENT)
        mock_post.return_value = _create_mock_response(200, response_data)

        provider = APIChatProvider(TEST_API_URL)
        message = [HumanMessage(content="Hello")]
        result = provider.send(message)

        assert isinstance(result, BaseMessage)
        mock_post.assert_called_once()

    @patch("quick_llm.chat.api_chat_provider.requests.post")
    def test_send_with_ai_message(self, mock_post):
        """Test sending an AIMessage"""
        response_data = _create_message_json(TEST_RESPONSE_CONTENT)
        mock_post.return_value = _create_mock_response(200, response_data)

        provider = APIChatProvider(TEST_API_URL)
        message = [AIMessage(content="Previous response")]
        result = provider.send(message)

        assert isinstance(result, BaseMessage)
        mock_post.assert_called_once()

    @patch("quick_llm.chat.api_chat_provider.requests.post")
    def test_send_with_message_list(self, mock_post):
        """Test sending a list of mixed message types"""
        response_data = _create_message_json(TEST_RESPONSE_CONTENT)
        mock_post.return_value = _create_mock_response(200, response_data)

        provider = APIChatProvider(TEST_API_URL)
        messages = [
            HumanMessage(content="Question 1"),
            AIMessage(content="Answer 1"),
            HumanMessage(content="Question 2"),
        ]
        result = provider.send(messages)

        assert isinstance(result, BaseMessage)
        mock_post.assert_called_once()

        # Verify all messages were sent
        call_args = mock_post.call_args
        request_data = call_args.kwargs["json"]
        assert len(request_data["messages"]) == 3


class TestRealWorldScenarios:
    """Test real-world usage scenarios"""

    @patch("quick_llm.chat.api_chat_provider.requests.post")
    def test_conversation_flow(self, mock_post):
        """Test a multi-turn conversation flow"""
        # Setup different responses for each call
        responses = [
            _create_message_json("Hello! How can I help you?"),
            _create_message_json("That's a great question!"),
            _create_message_json("You're welcome!"),
        ]

        mock_post.side_effect = [_create_mock_response(200, r) for r in responses]

        provider = APIChatProvider(TEST_API_URL)

        # Simulate conversation
        result1 = provider.send([HumanMessage(content="Hi")])
        assert isinstance(result1, BaseMessage)

        result2 = provider.send([HumanMessage(content="Can you help me?")])
        assert isinstance(result2, BaseMessage)

        result3 = provider.send([HumanMessage(content="Thank you")])
        assert isinstance(result3, BaseMessage)

        # Verify three requests were made
        assert mock_post.call_count == 3

    @patch("quick_llm.chat.api_chat_provider.requests.post")
    def test_retry_after_failure(self, mock_post):
        """Test retrying after a failed request"""
        # First call fails, second succeeds
        mock_post.side_effect = [
            _create_mock_response(500, "Server Error"),
            _create_mock_response(200, _create_message_json("Success!")),
        ]

        provider = APIChatProvider(TEST_API_URL)
        message = [HumanMessage(content=TEST_INPUT)]

        # First attempt fails
        result1 = provider.send(message)
        assert "Failed to get a response" in result1.content

        # Second attempt succeeds
        result2 = provider.send(message)
        assert isinstance(result2, BaseMessage)

        # Verify both requests were made
        assert mock_post.call_count == 2

    @patch("quick_llm.chat.api_chat_provider.requests.post")
    def test_different_api_urls(self, mock_post):
        """Test using different API URLs for different providers"""
        response_data = _create_message_json(TEST_RESPONSE_CONTENT)
        mock_post.return_value = _create_mock_response(200, response_data)

        provider1 = APIChatProvider("http://api1.example.com/chat")
        provider2 = APIChatProvider("http://api2.example.com/chat")

        message = [HumanMessage(content=TEST_INPUT)]
        provider1.send(message)
        provider2.send(message)

        # Verify different URLs were called
        assert mock_post.call_count == 2
        call1_url = mock_post.call_args_list[0].args[0]
        call2_url = mock_post.call_args_list[1].args[0]
        assert call1_url != call2_url
