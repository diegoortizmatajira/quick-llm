<a id="chain_endpoints"></a>

# chain\_endpoints

Module for defining chain-related API endpoints.

<a id="chain_endpoints.GenerateRequest"></a>

## GenerateRequest Objects

```python
class GenerateRequest()
```

Data class representing a generate request.

<a id="chain_endpoints.GenerateResponse"></a>

## GenerateResponse Objects

```python
class GenerateResponse()
```

Data class representing a generate response.

<a id="chain_endpoints.ChatRequest"></a>

## ChatRequest Objects

```python
class ChatRequest(BaseModel)
```

Represents a request to the chat API endpoint. It contains a list of the
latest messages exchanged between the user and the AI

<a id="chain_endpoints.ChatRequest.from_chat_input"></a>

#### from\_chat\_input

```python
@staticmethod
def from_chat_input(chat_input: ChatInputType) -> "ChatRequest"
```

Create a ChatRequest from ChatInputType.

**Arguments**:

- `chat_input` - The input messages in ChatInputType format.

**Returns**:

  A ChatRequest instance containing the messages.

<a id="chain_endpoints.ChatResponse"></a>

## ChatResponse Objects

```python
class ChatResponse()
```

Represents the response from the chat API, containing the generated message from the AI Assistant.

<a id="chain_endpoints.ChainEndpoints"></a>

## ChainEndpoints Objects

```python
class ChainEndpoints(Generic[ChainOutputVar])
```

A class representing endpoints for chains in a FastAPI application.

This class facilitates the creation and management of API endpoints linked
to `Runnable` chain functions. It provides the ability to define chat and generate
endpoints, including their streaming counterparts, and integrates them into
the FastAPI app.

**Attributes**:

- `chain` - A `Runnable` instance or a callable for initializing the chain.
- `app` - The FastAPI app instance.

<a id="chain_endpoints.ChainEndpoints.with_chat_endpoint"></a>

#### with\_chat\_endpoint

```python
@overload
def with_chat_endpoint(
        *,
        endpoint: str | None = "/api/chat",
        stream_endpoint: str | None = None,
        input_transformer: ChatInputTransformer | None = None,
        output_transformer: ChatOutputTransformer | None = None) -> Self
```

Add a chat endpoint to the chain.

This method allows the configuration of an endpoint for chat interactions
and optionally a streaming endpoint for real-time interaction. It supports
either providing a chat provider or configuring input/output transformers.

**Arguments**:

- `endpoint` - The endpoint path for the chat functionality (default: "api/chat").
- `stream_endpoint` - The optional endpoint path for streaming chat responses.
- `input_transformer` - An optional input transformer for modifying chat input.
- `output_transformer` - An optional output transformer for modifying chat output.
  

**Returns**:

  The ChainEndpoints instance with the chat endpoint configured.

<a id="chain_endpoints.ChainEndpoints.with_chat_endpoint"></a>

#### with\_chat\_endpoint

```python
@overload
def with_chat_endpoint(*,
                       endpoint: str | None = "/api/chat",
                       stream_endpoint: str | None = None,
                       chat_provider: ChainChatProvider | None = None) -> Self
```

Add a chat endpoint to the chain.

This method allows the configuration of an endpoint for chat interactions
and optionally a streaming endpoint for real-time interaction. It supports
either providing a chat provider or configuring input/output transformers.

**Arguments**:

- `endpoint` - The endpoint path for the chat functionality (default: "api/chat").
- `stream_endpoint` - The optional endpoint path for streaming chat responses.
- `chat_provider` - An optional pre-configured chat provider to use.
  

**Returns**:

  The ChainEndpoints instance with the chat endpoint configured.

<a id="chain_endpoints.ChainEndpoints.with_chat_endpoint"></a>

#### with\_chat\_endpoint

```python
def with_chat_endpoint(
        *,
        endpoint: str | None = "/api/chat",
        stream_endpoint: str | None = None,
        chat_provider: ChainChatProvider | None = None,
        input_transformer: ChatInputTransformer | None = None,
        output_transformer: ChatOutputTransformer | None = None) -> Self
```

Add a chat endpoint to the chain.

This method allows the configuration of an endpoint for chat interactions
and optionally a streaming endpoint for real-time interaction. It supports
either providing a chat provider or configuring input/output transformers.

**Arguments**:

- `endpoint` - The endpoint path for the chat functionality (default: "api/chat").
- `stream_endpoint` - The optional endpoint path for streaming chat responses.
- `chat_provider` - An optional pre-configured chat provider to use.
- `input_transformer` - An optional input transformer for modifying chat input.
- `output_transformer` - An optional output transformer for modifying chat output.
  

**Returns**:

  The ChainEndpoints instance with the chat endpoint configured.

<a id="chain_endpoints.ChainEndpoints.with_generate_endpoint"></a>

#### with\_generate\_endpoint

```python
def with_generate_endpoint(*,
                           endpoint: str | None = "/api/generate",
                           stream_endpoint: str | None = None) -> Self
```

Add a generate endpoint to the chain.

This method allows the configuration of an endpoint for generating
results from the chain. It supports both standard and streaming endpoints.

**Arguments**:

- `endpoint` - The endpoint path for the generate functionality (default: "api/generate").
- `stream_endpoint` - The optional endpoint path for streaming generate responses.
  

**Returns**:

  The ChainEndpoints instance with the generate endpoint configured.

<a id="chain_endpoints.ChainEndpoints.with_defaults"></a>

#### with\_defaults

```python
def with_defaults() -> Self
```

Add default endpoints to the chain.

**Returns**:

  The ChainEndpoints instance with default endpoints added.

<a id="chain_endpoints.ChainEndpoints.build"></a>

#### build

```python
def build() -> None
```

Build and register the endpoints with the FastAPI app.

<a id="chain_endpoints.ChainEndpoints.chain"></a>

#### chain

```python
@property
def chain() -> Runnable[ChainInputType, ChainOutputVar]
```

Get the chain instance.

**Returns**:

  The chain instance.

<a id="chain_endpoints.ChainEndpoints.serve_generate"></a>

#### serve\_generate

```python
def serve_generate(
    request: GenerateRequest[ChainInputType]
) -> GenerateResponse[ChainOutputVar]
```

Serve a generate request.

**Arguments**:

- `request` - The generate request containing the prompt.
  

**Returns**:

  The generate response containing the generated output.

<a id="chain_endpoints.ChainEndpoints.serve_generate_streaming"></a>

#### serve\_generate\_streaming

```python
def serve_generate_streaming(
        request: GenerateRequest[ChainInputType]) -> StreamingResponse
```

Serve a generate request with a streaming response.

This method handles requests to generate output in a streaming fashion.
It converts the output from the chain into JSONL (JSON Lines) format and
streams it as a response.

**Arguments**:

- `request` - The generate request containing the prompt.
  

**Returns**:

  A StreamingResponse where each item is a JSON-serialized
  GenerateResponse object, streamed to the client.

<a id="chain_endpoints.ChainEndpoints.serve_chat"></a>

#### serve\_chat

```python
def serve_chat(request: ChatRequest) -> ChatResponse[BaseMessage]
```

Serve a chat request.

This method processes a chat request by sending the received messages
to the configured chat provider and returning the generated response.

**Arguments**:

- `request` - The chat request containing the list of messages exchanged
  between the user and the AI.
  

**Returns**:

  A ChatResponse object containing the generated message and its creation timestamp.

<a id="chain_endpoints.ChainEndpoints.serve_chat_streaming"></a>

#### serve\_chat\_streaming

```python
def serve_chat_streaming(request: ChatRequest) -> StreamingResponse
```

Serve a chat request with a streaming response.

This method handles chat requests to generate output in a streaming fashion.
It converts the responses from the chat provider into JSONL (JSON Lines) format
and streams them as a response.

**Arguments**:

- `request` - The chat request containing the messages.
  

**Returns**:

  A StreamingResponse where each item is a JSON-serialized ChatResponse object,
  streamed to the client.

