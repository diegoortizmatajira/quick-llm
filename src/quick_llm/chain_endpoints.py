"""Module for defining chain-related API endpoints."""

from datetime import datetime, timezone
from typing import Callable, Generic, Iterator, Self, overload
from fastapi import APIRouter, FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from quick_llm.chain_chat_provider import (
    ChainChatProvider,
    ChatInputTransformer,
    ChatOutputTransformer,
)
from quick_llm.type_definitions import ChainInputType, ChainOutputVar


class GenerateRequest[T](BaseModel):
    """Data class representing a generate request."""

    prompt: T


class GenerateResponse[T](BaseModel):
    """Data class representing a generate response."""

    response: T
    created_at: datetime


class ChatRequest(BaseModel):
    """
    Represents a request to the chat API endpoint. It contains a list of the
    latest messages exchanged between the user and the AI
    """

    messages: list[BaseMessage]


class ChatResponse[MessageTypeVar: BaseMessage](BaseModel):
    """
    Represents the response from the chat API, containing the generated message from the AI Assistant.
    """

    message: MessageTypeVar
    created_at: datetime


class ChainEndpoints(Generic[ChainOutputVar]):
    """
    A class representing endpoints for chains in a FastAPI application.

    This class facilitates the creation and management of API endpoints linked
    to `Runnable` chain functions. It provides the ability to define chat and generate
    endpoints, including their streaming counterparts, and integrates them into
    the FastAPI app.

    Attributes:
        chain: A `Runnable` instance or a callable for initializing the chain.
        app: The FastAPI app instance.
    """

    def __init__(
        self,
        app: FastAPI,
        chain: Runnable[ChainInputType, ChainOutputVar]
        | Callable[[], Runnable[ChainInputType, ChainOutputVar]],
    ):
        self.__chain = chain
        self.__endpoints_registered = False
        self.__chat_provider: ChainChatProvider | None = None
        self.__app = app
        self.__router = APIRouter()

    @overload
    def with_chat_endpoint(
        self,
        *,
        endpoint: str | None = "api/chat",
        stream_endpoint: str | None = None,
        input_transformer: ChatInputTransformer | None = None,
        output_transformer: ChatOutputTransformer | None = None,
    ) -> Self:
        """Add a chat endpoint to the chain.

        This method allows the configuration of an endpoint for chat interactions
        and optionally a streaming endpoint for real-time interaction. It supports
        either providing a chat provider or configuring input/output transformers.

        Args:
            endpoint: The endpoint path for the chat functionality (default: "api/chat").
            stream_endpoint: The optional endpoint path for streaming chat responses.
            input_transformer: An optional input transformer for modifying chat input.
            output_transformer: An optional output transformer for modifying chat output.

        Returns:
            The ChainEndpoints instance with the chat endpoint configured.
        """

    @overload
    def with_chat_endpoint(
        self,
        *,
        endpoint: str | None = "api/chat",
        stream_endpoint: str | None = None,
        chat_provider: ChainChatProvider | None = None,
    ) -> Self:
        """Add a chat endpoint to the chain.

        This method allows the configuration of an endpoint for chat interactions
        and optionally a streaming endpoint for real-time interaction. It supports
        either providing a chat provider or configuring input/output transformers.

        Args:
            endpoint: The endpoint path for the chat functionality (default: "api/chat").
            stream_endpoint: The optional endpoint path for streaming chat responses.
            chat_provider: An optional pre-configured chat provider to use.

        Returns:
            The ChainEndpoints instance with the chat endpoint configured.
        """

    def with_chat_endpoint(
        self,
        *,
        endpoint: str | None = "api/chat",
        stream_endpoint: str | None = None,
        chat_provider: ChainChatProvider | None = None,
        input_transformer: ChatInputTransformer | None = None,
        output_transformer: ChatOutputTransformer | None = None,
    ) -> Self:
        """Add a chat endpoint to the chain.

        This method allows the configuration of an endpoint for chat interactions
        and optionally a streaming endpoint for real-time interaction. It supports
        either providing a chat provider or configuring input/output transformers.

        Args:
            endpoint: The endpoint path for the chat functionality (default: "api/chat").
            stream_endpoint: The optional endpoint path for streaming chat responses.
            chat_provider: An optional pre-configured chat provider to use.
            input_transformer: An optional input transformer for modifying chat input.
            output_transformer: An optional output transformer for modifying chat output.

        Returns:
            The ChainEndpoints instance with the chat endpoint configured.
        """
        if endpoint == stream_endpoint:
            raise ValueError("Endpoint and stream_endpoint cannot be the same.")

        chat_endpoints_registered = False
        if endpoint:
            self.__router.add_api_route(
                endpoint,
                self.serve_chat,
                methods=["POST"],
            )
            chat_endpoints_registered = True
        if stream_endpoint:
            self.__router.add_api_route(
                stream_endpoint,
                self.serve_chat_streaming,
                methods=["POST"],
                response_class=StreamingResponse,
            )
            chat_endpoints_registered = True
        self.__endpoints_registered = (
            self.__endpoints_registered or chat_endpoints_registered
        )
        if chat_endpoints_registered:
            if not chat_provider:
                chat_provider = ChainChatProvider(
                    chain=self.__chain,
                    input_transformer=input_transformer,
                    output_transformer=output_transformer,
                )
            self.__chat_provider = chat_provider
        return self

    def with_generate_endpoint(
        self,
        *,
        endpoint: str | None = "api/generate",
        stream_endpoint: str | None = None,
    ) -> Self:
        """Add a generate endpoint to the chain.

        This method allows the configuration of an endpoint for generating
        results from the chain. It supports both standard and streaming endpoints.

        Args:
            endpoint: The endpoint path for the generate functionality (default: "api/generate").
            stream_endpoint: The optional endpoint path for streaming generate responses.

        Returns:
            The ChainEndpoints instance with the generate endpoint configured.
        """
        if endpoint:
            self.__router.add_api_route(
                endpoint,
                self.serve_generate,
                methods=["POST"],
                response_model=GenerateResponse[ChainOutputVar],
            )
            self.__endpoints_registered = True
        if stream_endpoint:
            self.__router.add_api_route(
                stream_endpoint,
                self.serve_generate_streaming,
                methods=["POST"],
                response_class=StreamingResponse,
                response_model=GenerateResponse[ChainOutputVar],
            )
            self.__endpoints_registered = True
        return self

    def with_defaults(self) -> Self:
        """Add default endpoints to the chain.

        Returns:
            The ChainEndpoints instance with default endpoints added.
        """
        return self.with_generate_endpoint().with_chat_endpoint()

    def build(self) -> None:
        """Build and register the endpoints with the FastAPI app."""
        if not self.__endpoints_registered:
            # If no endpoints are defined, add default ones.
            self.with_defaults()
        self.__app.include_router(self.__router)

    @property
    def chain(self) -> Runnable[ChainInputType, ChainOutputVar]:
        """Get the chain instance.

        Returns:
            The chain instance.
        """
        return self.__chain() if callable(self.__chain) else self.__chain

    def serve_generate(
        self, request: GenerateRequest[ChainInputType]
    ) -> GenerateResponse[ChainOutputVar]:
        """Serve a generate request.

        Args:
            request: The generate request containing the prompt.

        Returns:
            The generate response containing the generated output.
        """
        output = self.chain.invoke(request.prompt)
        return GenerateResponse(
            response=output,
            created_at=datetime.now(timezone.utc),
        )

    def serve_generate_streaming(
        self, request: GenerateRequest[ChainInputType]
    ) -> StreamingResponse:
        """Serve a generate request with a streaming response.

        This method handles requests to generate output in a streaming fashion.
        It converts the output from the chain into JSONL (JSON Lines) format and
        streams it as a response.

        Args:
            request: The generate request containing the prompt.

        Returns:
            A StreamingResponse where each item is a JSON-serialized
            GenerateResponse object, streamed to the client.
        """

        def transformer(output_stream: Iterator[ChainOutputVar]):
            for output in output_stream:
                yield (
                    GenerateResponse(
                        response=output,
                        created_at=datetime.now(timezone.utc),
                    ).model_dump_json()
                    + "\n"
                )

        result = transformer(self.chain.stream(request.prompt))
        return StreamingResponse(result, media_type="application/x-ndjson")

    def serve_chat(self, request: ChatRequest) -> ChatResponse[BaseMessage]:
        """Serve a chat request.

        This method processes a chat request by sending the received messages
        to the configured chat provider and returning the generated response.

        Args:
            request: The chat request containing the list of messages exchanged 
                     between the user and the AI.

        Returns:
            A ChatResponse object containing the generated message and its creation timestamp.
        """
        if not self.__chat_provider:
            raise ValueError("Chat provider is not configured.")
        full_message = self.__chat_provider.send(request.messages)
        return ChatResponse(
            message=full_message,
            created_at=datetime.now(timezone.utc),
        )

    def serve_chat_streaming(self, request: ChatRequest) -> StreamingResponse:
        """Serve a chat request with a streaming response.

        This method handles chat requests to generate output in a streaming fashion.
        It converts the responses from the chat provider into JSONL (JSON Lines) format
        and streams them as a response.

        Args:
            request: The chat request containing the messages.

        Returns:
            A StreamingResponse where each item is a JSON-serialized ChatResponse object,
            streamed to the client.
        """
        if not self.__chat_provider:
            raise ValueError("Chat provider is not configured.")

        def transformer(response: Iterator[BaseMessage]) -> Iterator[str]:
            for item in response:
                chunk_response = ChatResponse(
                    message=item, created_at=datetime.now(timezone.utc)
                )

                yield chunk_response.model_dump_json() + "\n"

        result = transformer(self.__chat_provider.send_stream(request.messages))

        return StreamingResponse(result, media_type="application/x-ndjson")
