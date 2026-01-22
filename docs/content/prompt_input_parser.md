<a id="prompt_input_parser"></a>

# prompt\_input\_parser

Module to parse prompt inputs into a standardized dictionary format.

<a id="prompt_input_parser.PromptInputParser"></a>

## PromptInputParser Objects

```python
class PromptInputParser(RunnableGenerator[ChainInputType, dict])
```

A parser class that converts prompt input types into a unified dictionary format.

This class inherits from RunnableGenerator and provides methods to transform
various input types (e.g., BaseModel, dictionaries, or other values) into
dictionary format, offering synchronous and asynchronous parsing capabilities.

**Arguments**:

- `prompt_input_param` _str_ - The parameter name to use when transforming non-dict inputs into a dictionary.

<a id="prompt_input_parser.PromptInputParser.transform_value"></a>

#### transform\_value

```python
def transform_value(value: ChainInputType) -> dict
```

Transforms the input value into a dictionary format.

<a id="prompt_input_parser.PromptInputParser.input_parser"></a>

#### input\_parser

```python
def input_parser(input_value: Iterator[ChainInputType]) -> Iterator[dict]
```

Parses any non dictionary value into a dictionary

<a id="prompt_input_parser.PromptInputParser.ainput_parser"></a>

#### ainput\_parser

```python
async def ainput_parser(
        input_value: AsyncIterator[ChainInputType]) -> AsyncIterator[dict]
```

Parses any non dictionary value into a dictionary

