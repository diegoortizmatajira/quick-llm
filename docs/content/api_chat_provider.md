<a id="chat.api_chat_provider"></a>

# chat.api\_chat\_provider

API-based Chat Provider Module.

<a id="chat.api_chat_provider.APIChatProvider"></a>

## APIChatProvider Objects

```python
class APIChatProvider(ChatProvider)
```

A chat provider that interacts with an external API for processing chat input.

This class uses an API endpoint specified by the `url` parameter to send chat
input and retrieve responses. It wraps both synchronous and asynchronous message
sending, as well as streaming capabilities.

**Attributes**:

- `url` _str_ - The URL of the API endpoint.
- `logger` _logging.Logger_ - Logger instance for logging activity of the class.

