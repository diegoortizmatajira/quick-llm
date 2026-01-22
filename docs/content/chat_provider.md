<a id="chat.chat_provider"></a>

# chat.chat\_provider

Defines the abstract base class for chat provider adapters.

<a id="chat.chat_provider.ChatProvider"></a>

## ChatProvider Objects

```python
class ChatProvider()
```

Represents an abstract class that defines the expected behavior of a chat provider adapter.

<a id="chat.chat_provider.ChatProvider.send"></a>

#### send

```python
@abstractmethod
def send(message: ChatInputType) -> BaseMessage
```

Send a message synchronously and return the response.

**Arguments**:

- `message` _ChatInputType_ - The input message(s) to send.
  

**Returns**:

- `BaseMessage` - The response from the chat provider.

<a id="chat.chat_provider.ChatProvider.send_async"></a>

#### send\_async

```python
@abstractmethod
async def send_async(message: ChatInputType) -> BaseMessage
```

Send a message asynchronously and return the response.

**Arguments**:

- `message` _ChatInputType_ - The input message(s) to send.
  

**Returns**:

- `BaseMessage` - The asynchronous response from the chat provider.

<a id="chat.chat_provider.ChatProvider.send_stream"></a>

#### send\_stream

```python
@abstractmethod
def send_stream(message: ChatInputType) -> Iterator[BaseMessage]
```

Send a message as a stream and yield responses.

**Arguments**:

- `message` _ChatInputType_ - The input message(s) to send.
  

**Yields**:

- `Iterator[BaseMessage]` - An iterator that yields responses from the chat provider.

