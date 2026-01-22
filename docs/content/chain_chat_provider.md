<a id="chat.chain_chat_provider"></a>

# chat.chain\_chat\_provider

Chat provider for communication with a chain.

<a id="chat.chain_chat_provider.ChainChatProvider"></a>

## ChainChatProvider Objects

```python
class ChainChatProvider()
```

A chat provider that facilitates communication with a chain.

This class supports synchronous, asynchronous, and streaming messaging
with the chain, using optional input and output transformers for
pre-processing and post-processing of data.

**Arguments**:

- `chain` - A Runnable instance or a factory function that
  returns a Runnable instance, representing the chain to
  communicate with.
- `input_transformer` - An optional callable that transforms
  inputs before passing them to the chain.
  Defaults to the `default_input_transformer`.
- `output_transformer` - An optional callable that transforms
  outputs received from the chain.
  Defaults to the `default_output_transformer`.

<a id="chat.chain_chat_provider.ChainChatProvider.default_input_transformer"></a>

#### default\_input\_transformer

```python
def default_input_transformer(input_value: ChatInputType) -> ChainInputType
```

Default input transformer that assumes input is already MessageLike.

**Arguments**:

- `input` - The chain input.
  

**Returns**:

  The input as MessageLike.

<a id="chat.chain_chat_provider.ChainChatProvider.default_output_transformer"></a>

#### default\_output\_transformer

```python
def default_output_transformer(output_value: ChainOutputVar) -> BaseMessage
```

Default output transformer that assumes output is already MessageLike.

**Arguments**:

- `output` - The chain output.
  

**Returns**:

  The output as MessageLike.

<a id="chat.chain_chat_provider.ChainChatProvider.send"></a>

#### send

```python
@override
def send(message: ChatInputType) -> BaseMessage
```

Send a message to the chain and retrieve a response.

**Arguments**:

- `message` - The input message to be transformed and sent to the chain.
  

**Returns**:

  The processed output message from the chain.

<a id="chat.chain_chat_provider.ChainChatProvider.send_async"></a>

#### send\_async

```python
@override
async def send_async(message: ChatInputType) -> BaseMessage
```

Send a message to the chain and retrieve a response.

**Arguments**:

- `message` - The input message to be transformed and sent to the chain.
  

**Returns**:

  The processed output message from the chain.

<a id="chat.chain_chat_provider.ChainChatProvider.send_stream"></a>

#### send\_stream

```python
@override
def send_stream(message: ChatInputType) -> Iterator[BaseMessage]
```

Send a message to the chain and retrieve a response.

**Arguments**:

- `message` - The input message to be transformed and sent to the chain.
  

**Returns**:

  The processed output message from the chain.

