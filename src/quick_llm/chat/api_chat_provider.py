"""API-based Chat Provider Module."""

import logging
from typing import Iterator, override

import requests
from langchain_core.messages import AIMessage, BaseMessage

from quick_llm.chain_endpoints import ChatRequest

from .chat_provider import ChatInputType, ChatProvider


class APIChatProvider(ChatProvider):
    """
    A chat provider that interacts with an external API for processing chat input.

    This class uses an API endpoint specified by the `url` parameter to send chat
    input and retrieve responses. It wraps both synchronous and asynchronous message
    sending, as well as streaming capabilities.

    Attributes:
        url (str): The URL of the API endpoint.
        logger (logging.Logger): Logger instance for logging activity of the class.
    """

    def __init__(self, url: str, timeout: int | None = None):
        super().__init__()
        self.url = url
        self.logger = logging.getLogger(__name__)
        self.logger.info("Configuring chat provider to use API with url: %s", url)
        self.timeout = timeout

    @override
    def send(self, message: ChatInputType) -> BaseMessage:
        request = ChatRequest.from_chat_input(message)
        json_content = request.model_dump()
        self.logger.debug("Sending message to API. Content: %s", json_content)
        response = requests.post(self.url, json=json_content, timeout=self.timeout)
        if response.status_code != 200:
            self.logger.error(
                "Error sending request to API.\nStatus code: %s,\nContent: %s",
                response.status_code,
                response.content.decode("utf-8"),
            )
            return AIMessage("Failed to get a response from the server")
        self.logger.debug("Response from the API. Content: %s", response.content)
        return BaseMessage.model_validate(response.content)

    @override
    async def send_async(self, message: ChatInputType) -> BaseMessage:
        return self.send(message)  # For simplicity, using the sync version

    @override
    def send_stream(self, message: ChatInputType) -> Iterator[BaseMessage]:
        yield self.send(message)  # For simplicity, using the sync version
