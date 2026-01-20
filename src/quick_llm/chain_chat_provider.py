"""Chat provider for communication with a chain."""

from typing import Any, Callable, Iterable, Iterator

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompts.chat import MessageLike
from langchain_core.runnables import Runnable

from quick_llm.type_definitions import ChainInputType


ChatInputType = MessageLike | Iterable[MessageLike]
ChatInputTransformer = Callable[[ChatInputType], ChainInputType]
ChatOutputTransformer = Callable[[Any], BaseMessage]


class ChainChatProvider[ChainOutputVar]:
    """A chat provider that facilitates communication with a chain.

    This class supports synchronous, asynchronous, and streaming messaging
    with the chain, using optional input and output transformers for
    pre-processing and post-processing of data.

    Args:
        chain: A Runnable instance or a factory function that
            returns a Runnable instance, representing the chain to
            communicate with.
        input_transformer: An optional callable that transforms
            inputs before passing them to the chain.
            Defaults to the `default_input_transformer`.
        output_transformer: An optional callable that transforms
            outputs received from the chain.
            Defaults to the `default_output_transformer`.
    """

    def __init__(
        self,
        chain: Runnable[ChainInputType, ChainOutputVar]
        | Callable[[], Runnable[ChainInputType, ChainOutputVar]],
        *,
        input_transformer: ChatInputTransformer | None = None,
        output_transformer: ChatOutputTransformer | None = None,
    ):
        # Stores a factory function to get the chain instance
        self.__chain_factory = chain if callable(chain) else lambda: chain
        self.__input_transformer = input_transformer or self.default_input_transformer
        self.__output_transformer = (
            output_transformer or self.default_output_transformer
        )

    def default_input_transformer(self, input_value: ChatInputType) -> ChainInputType:
        """Default input transformer that assumes input is already MessageLike.

        Args:
            input: The chain input.

        Returns:
            The input as MessageLike.
        """

        def msg_to_str(msg: MessageLike) -> str:
            return msg.pretty_repr()

        if isinstance(input_value, Iterable) and not isinstance(
            input_value, (str, BaseMessage)
        ):
            return "\n".join(msg_to_str(msg) for msg in input_value)  # type: ignore
        return msg_to_str(input_value)

    def default_output_transformer(self, output_value: ChainOutputVar) -> BaseMessage:
        """Default output transformer that assumes output is already MessageLike.

        Args:
            output: The chain output.

        Returns:
            The output as MessageLike.
        """
        if isinstance(output_value, list):
            return AIMessage(content=[str(item) for item in output_value])
        if isinstance(output_value, str):
            return AIMessage(content=output_value)
        if isinstance(output_value, BaseMessage):
            return output_value
        return AIMessage(content=str(output_value))

    def send(self, message: ChatInputType) -> BaseMessage:
        """Send a message to the chain and retrieve a response.

        Args:
            message: The input message to be transformed and sent to the chain.

        Returns:
            The processed output message from the chain.
        """
        chain = self.__chain_factory()
        formatted_input = self.__input_transformer(message)
        chain_output = chain.invoke(formatted_input)
        formatted_output = self.__output_transformer(chain_output)
        return formatted_output

    async def send_async(self, message: ChatInputType) -> BaseMessage:
        """Send a message to the chain and retrieve a response.

        Args:
            message: The input message to be transformed and sent to the chain.

        Returns:
            The processed output message from the chain.
        """
        chain = self.__chain_factory()
        formatted_input = self.__input_transformer(message)
        chain_output = await chain.ainvoke(formatted_input)
        formatted_output = self.__output_transformer(chain_output)
        return formatted_output

    def send_stream(self, message: ChatInputType) -> Iterator[BaseMessage]:
        """Send a message to the chain and retrieve a response.

        Args:
            message: The input message to be transformed and sent to the chain.

        Returns:
            The processed output message from the chain.
        """
        chain = self.__chain_factory()
        formatted_input = self.__input_transformer(message)
        stream = chain.stream(formatted_input)
        for chain_output in stream:
            formatted_output = self.__output_transformer(chain_output)
            yield formatted_output
