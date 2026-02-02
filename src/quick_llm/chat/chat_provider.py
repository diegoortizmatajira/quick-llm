"""Defines the abstract base class for chat provider adapters."""

from abc import abstractmethod
from typing import Any, Callable, Iterable, Iterator

from langchain_core.messages import BaseMessage
from langchain_core.prompts.chat import MessageLike

from ..support import ChainInputType


ChatInputType = MessageLike | Iterable[MessageLike]
ChatInputTransformer = Callable[[ChatInputType], ChainInputType]
ChatOutputTransformer = Callable[[Any], BaseMessage]


class ChatProvider[ABC]:
    """
    Represents an abstract class that defines the expected behavior of a chat provider adapter.
    """

    @abstractmethod
    def send(self, message: ChatInputType) -> BaseMessage:
        """Send a message synchronously and return the response.

        Args:
            message (ChatInputType): The input message(s) to send.

        Returns:
            BaseMessage: The response from the chat provider.
        """

    @abstractmethod
    async def send_async(self, message: ChatInputType) -> BaseMessage:
        """Send a message asynchronously and return the response.

        Args:
            message (ChatInputType): The input message(s) to send.

        Returns:
            BaseMessage: The asynchronous response from the chat provider.
        """

    @abstractmethod
    def send_stream(self, message: ChatInputType) -> Iterator[BaseMessage]:
        """Send a message as a stream and yield responses.

        Args:
            message (ChatInputType): The input message(s) to send.

        Yields:
            Iterator[BaseMessage]: An iterator that yields responses from the chat provider.
        """
