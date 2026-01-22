<a id="chain_factory"></a>

# chain\_factory

Factory class for managing language model instances.

<a id="chain_factory.ChainFactory"></a>

## ChainFactory Objects

```python
class ChainFactory(Generic[ChainOutputVar])
```

Factory class for managing language model instances.

<a id="chain_factory.ChainFactory.for_json_model"></a>

#### for\_json\_model

```python
@staticmethod
def for_json_model(
        json_model: type[BaseModel]) -> "ChainFactory[dict[str, object]]"
```

Creates a ChainFactory instance based on a given JSON model.

**Arguments**:

- `json_model`: A Pydantic BaseModel class that will be used to interpret JSON outputs.

**Returns**:

A ChainFactory instance configured to use the provided JSON model.

<a id="chain_factory.ChainFactory.for_rag_with_sources"></a>

#### for\_rag\_with\_sources

```python
@staticmethod
def for_rag_with_sources(
    json_model: type[BaseModel] | None = None
) -> "ChainFactory[dict[str, object]]"
```

Creates a ChainFactory instance based on a given JSON model.

**Arguments**:

- `json_model`: A Pydantic BaseModel class that will be used to interpret JSON outputs.

**Returns**:

A ChainFactory instance configured to use the provided JSON model.

<a id="chain_factory.ChainFactory.default_cleaner_function"></a>

#### default\_cleaner\_function

```python
def default_cleaner_function(text: str) -> str
```

Default function to clean the output text.

**Arguments**:

- `text`: The text to be cleaned.

**Returns**:

The cleaned text.

<a id="chain_factory.ChainFactory.default_context_formatter"></a>

#### default\_context\_formatter

```python
def default_context_formatter(documents: list[Document]) -> str
```

Default function to format context from a list of documents.

**Arguments**:

- `documents`: A list of Document instances.

**Returns**:

A formatted string representing the context.

<a id="chain_factory.ChainFactory.default_references_formatter"></a>

#### default\_references\_formatter

```python
def default_references_formatter(documents: list[Document]) -> str
```

Default function to format references from a list of documents.

**Arguments**:

- `documents`: A list of Document instances.

**Returns**:

A formatted string representing the references.

<a id="chain_factory.ChainFactory.get_readable_value"></a>

#### get\_readable\_value

```python
@staticmethod
def get_readable_value(value: object) -> object
```

Converts the input object into a human-readable format.

**Arguments**:

- `value`: The object to be converted. This can be a BaseMessage, BaseModel, or other types.

**Returns**:

A human-readable representation of the object.

<a id="chain_factory.ChainFactory.passthrough_logger"></a>

#### passthrough\_logger

```python
def passthrough_logger(caption: str) -> Runnable[T, T]
```

Captures the outputs and logs it. It is included in the default implementation of `wrap_chain` method

<a id="chain_factory.ChainFactory.wrap"></a>

#### wrap

```python
def wrap(runnable: Runnable[Input, Output],
         caption: str) -> Runnable[Input, Output]
```

Wraps a runnable with detailed logging if enabled.

**Arguments**:

- `runnable`: The runnable to be wrapped.

**Returns**:

The wrapped runnable with logging if detailed logging is enabled.

<a id="chain_factory.ChainFactory.language_model"></a>

#### language\_model

```python
@property
def language_model() -> LanguageModelLike
```

Gets the language model instance.

**Returns**:

The current instance of BaseLanguageModel or None if not set.

<a id="chain_factory.ChainFactory.prompt_template"></a>

#### prompt\_template

```python
@property
def prompt_template() -> BasePromptTemplate[PromptValue]
```

Gets the prompt template instance.

**Returns**:

The current instance of PromptTemplate or None if not set.

<a id="chain_factory.ChainFactory.input_param"></a>

#### input\_param

```python
@property
def input_param() -> str
```

Gets the name of the input parameter.

**Returns**:

The name of the input parameter.

<a id="chain_factory.ChainFactory.format_instructions_param"></a>

#### format\_instructions\_param

```python
@property
def format_instructions_param() -> str
```

Gets the name of the format instructions parameter.

**Returns**:

The name of the format instructions parameter.

<a id="chain_factory.ChainFactory.input_transformer"></a>

#### input\_transformer

```python
@property
def input_transformer() -> Runnable[ChainInputType, dict]
```

Gets the input transformer instance.

**Returns**:

The current instance of Runnable for input transformation.

<a id="chain_factory.ChainFactory.additional_values_injector"></a>

#### additional\_values\_injector

```python
@property
def additional_values_injector() -> Runnable[dict, dict]
```

Provides a lambda function that injects additional values into the existing input dictionary.

This method creates a dictionary of additional values to be passed into the chain. If the JSON model
is being used and the output transformer is of the type JsonOutputParser, it adds format instructions
specific to the JSON model to the `additional_values` dictionary. The lambda function merges the
existing input dictionary with these additional values.

**Returns**:

A Runnable instance that injects additional values into the input dictionary.

<a id="chain_factory.ChainFactory.output_cleaner"></a>

#### output\_cleaner

```python
@property
def output_cleaner() -> Runnable[LanguageModelOutput, LanguageModelOutput]
```

This function is used to clean the output messages from invalid escape sequences.
It is included in the default implementation of chains to ensure the output is valid.

<a id="chain_factory.ChainFactory.output_transformer"></a>

#### output\_transformer

```python
@property
def output_transformer() -> Runnable[LanguageModelOutput, ChainOutputVar]
```

Gets the output transformer instance.

**Returns**:

The current instance of Runnable for output transformation.

<a id="chain_factory.ChainFactory.text_splitter"></a>

#### text\_splitter

```python
@property
def text_splitter() -> TextSplitter
```

Gets the text splitter instance.

**Returns**:

The current instance of TextSplitter.

<a id="chain_factory.ChainFactory.embeddings"></a>

#### embeddings

```python
@property
def embeddings() -> Embeddings
```

Gets the embeddings instance.

**Returns**:

The current instance of Embeddings.

<a id="chain_factory.ChainFactory.vector_store"></a>

#### vector\_store

```python
@property
def vector_store() -> VectorStore
```

Gets the vector store instance.

**Returns**:

The current instance of VectorStore.

<a id="chain_factory.ChainFactory.retriever"></a>

#### retriever

```python
@property
def retriever() -> RetrieverLike
```

Gets the retriever instance.

**Returns**:

The current instance of RetrieverLike.

<a id="chain_factory.ChainFactory.document_formatter"></a>

#### document\_formatter

```python
@property
def document_formatter() -> Runnable[list[Document], str]
```

Allows the context retrieval to be formatted as a string to be passed down to the prompt.

<a id="chain_factory.ChainFactory.final_answer_formatter"></a>

#### final\_answer\_formatter

```python
@property
def final_answer_formatter() -> Runnable[dict, str]
```

Returns the final answer formatted along with the source references in
a single string.

<a id="chain_factory.ChainFactory.answer_key"></a>

#### answer\_key

```python
@property
def answer_key() -> str
```

Gets the name of the answer key in the output.

**Returns**:

The name of the answer key.

<a id="chain_factory.ChainFactory.document_references_key"></a>

#### document\_references\_key

```python
@property
def document_references_key() -> str
```

Gets the name of the document references key in the output.

**Returns**:

The name of the document references key.

<a id="chain_factory.ChainFactory.use"></a>

#### use

```python
def use(visitor: Callable[[Self], None]) -> Self
```

Applies a visitor function to the ChainFactory instance.

**Arguments**:

- `visitor`: A callable that takes a ChainFactory instance and returns None.

**Returns**:

The ChainFactory instance for method chaining.

<a id="chain_factory.ChainFactory.use_detailed_logging"></a>

#### use\_detailed\_logging

```python
def use_detailed_logging(enable: bool = True) -> Self
```

Enables or disables detailed logging for the ChainFactory.

**Arguments**:

- `enable`: A boolean flag to enable or disable detailed logging. Defaults to True.

**Returns**:

The ChainFactory instance for method chaining.

<a id="chain_factory.ChainFactory.use_language_model"></a>

#### use\_language\_model

```python
def use_language_model(language_model: LanguageModelLike) -> Self
```

Sets the language model instance.

**Arguments**:

- `language_model`: An instance of BaseLanguageModel to set.

**Returns**:

The ChainFactory instance for method chaining.

<a id="chain_factory.ChainFactory.use_input_param"></a>

#### use\_input\_param

```python
def use_input_param(name: str = "input") -> Self
```

Sets the name of the input parameter.

**Arguments**:

- `name`: The name to set for the input parameter. Defaults to 'input'.

**Returns**:

The ChainFactory instance for method chaining.

<a id="chain_factory.ChainFactory.use_format_instructions_param"></a>

#### use\_format\_instructions\_param

```python
def use_format_instructions_param(name: str = "format_instructions") -> Self
```

Sets the name of the format instructions parameter.

**Arguments**:

- `name`: The name to set for the format instructions parameter.
Defaults to 'format_instructions'.

**Returns**:

The ChainFactory instance for method chaining.

<a id="chain_factory.ChainFactory.use_context_param"></a>

#### use\_context\_param

```python
def use_context_param(name: str = "context") -> Self
```

Sets the name of the context parameter.

**Arguments**:

- `name`: The name to set for the context parameter. Defaults to 'context'.

**Returns**:

The ChainFactory instance for method chaining.

<a id="chain_factory.ChainFactory.use_answer_key"></a>

#### use\_answer\_key

```python
def use_answer_key(name: str = "answer") -> Self
```

Sets the name of the answer key in the output.

**Arguments**:

- `name`: The name to set for the answer key. Defaults to 'answer'.

**Returns**:

The ChainFactory instance for method chaining.

<a id="chain_factory.ChainFactory.use_prompt_template"></a>

#### use\_prompt\_template

```python
@overload
def use_prompt_template(
        prompt_template: BasePromptTemplate[PromptValue]) -> Self
```

Sets the prompt template instance.

**Arguments**:

- `prompt_template`: An instance of PromptTemplate to set.

**Returns**:

The ChainFactory instance for method chaining.

<a id="chain_factory.ChainFactory.use_prompt_template"></a>

#### use\_prompt\_template

```python
@overload
def use_prompt_template(
        prompt_template: str,
        prompt_template_format: PromptTemplateFormat = "f-string",
        partial_variables: dict[str, str] | None = None) -> Self
```

Sets the prompt template instance from a string.

**Arguments**:

- `prompt_template`: A string representing the prompt template.
- `prompt_template_format`: The format of the prompt template string.
- `partial_variables`: A dictionary of partial variables for the prompt template.

**Returns**:

The ChainFactory instance for method chaining.

<a id="chain_factory.ChainFactory.use_prompt_template"></a>

#### use\_prompt\_template

```python
def use_prompt_template(
        prompt_template: str | BasePromptTemplate[PromptValue],
        prompt_template_format: PromptTemplateFormat = "f-string",
        partial_variables: dict[str, str] | None = None) -> Self
```

Sets the prompt template instance.

**Arguments**:

- `prompt_template`: An instance of PromptTemplate or a string representing
the prompt template.
- `prompt_template_format`: The format of the prompt template string.
- `partial_variables`: A dictionary of partial variables for the prompt template.

**Returns**:

The ChainFactory instance for method chaining.

<a id="chain_factory.ChainFactory.use_json_model"></a>

#### use\_json\_model

```python
def use_json_model(model: type[BaseModel]) -> Self
```

Sets the JSON model for output parsing.

**Arguments**:

- `model`: A Pydantic BaseModel class to parse the output into.

**Returns**:

The ChainFactory instance for method chaining.

<a id="chain_factory.ChainFactory.use_output_transformer"></a>

#### use\_output\_transformer

```python
@overload
def use_output_transformer(
        output_parser: Runnable[LanguageModelOutput, ChainOutputVar]) -> Self
```

Sets the output transformer instance.

**Arguments**:

- `output_parser`: An instance of Runnable for output transformation.
If None, a default StrOutputParser is used.

**Returns**:

The ChainFactory instance for method chaining.

<a id="chain_factory.ChainFactory.use_output_transformer"></a>

#### use\_output\_transformer

```python
@overload
def use_output_transformer(
        output_parser: Callable[[LanguageModelOutput],
                                ChainOutputVar]) -> Self
```

Sets the output transformer instance.

**Arguments**:

- `output_parser`: An instance of Callable for output transformation.
If None, a default StrOutputParser is used.

**Returns**:

The ChainFactory instance for method chaining.

<a id="chain_factory.ChainFactory.use_output_transformer"></a>

#### use\_output\_transformer

```python
def use_output_transformer(
    output_parser: Runnable[LanguageModelOutput, ChainOutputVar]
    | Callable[[LanguageModelOutput], ChainOutputVar]
) -> Self
```

Sets the output transformer instance.

**Arguments**:

- `output_parser`: An instance of Runnable for output transformation.
If None, a default StrOutputParser is used.

**Returns**:

The ChainFactory instance for method chaining.

<a id="chain_factory.ChainFactory.use_custom_output_cleaner"></a>

#### use\_custom\_output\_cleaner

```python
def use_custom_output_cleaner(cleaner_function: Callable[[str], str]) -> Self
```

Sets a custom output cleaner function.

**Arguments**:

- `cleaner_function`: A callable that takes a string and returns a cleaned string.

**Returns**:

The ChainFactory instance for method chaining.

<a id="chain_factory.ChainFactory.use_custom_context_formatter"></a>

#### use\_custom\_context\_formatter

```python
def use_custom_context_formatter(
        formatter_function: Callable[[list[Document]], str]) -> Self
```

Sets a custom context formatter function.

**Arguments**:

- `formatter_function`: A callable that takes a list of Document instances
and returns a formatted string.

**Returns**:

The ChainFactory instance for method chaining.

<a id="chain_factory.ChainFactory.use_custom_retrieval_query_builder"></a>

#### use\_custom\_retrieval\_query\_builder

```python
def use_custom_retrieval_query_builder(
        query_builder_function: Callable[[dict], str]) -> Self
```

Sets a custom retrieval query builder function.

**Arguments**:

- `query_builder_function`: A callable that takes a dictionary of input values
and returns a query string.

**Returns**:

The ChainFactory instance for method chaining.

<a id="chain_factory.ChainFactory.use_text_splitter"></a>

#### use\_text\_splitter

```python
def use_text_splitter(text_splitter: TextSplitter) -> Self
```

Sets the text splitter instance.

**Arguments**:

- `text_splitter`: An instance of TextSplitter to set.

**Returns**:

The ChainFactory instance for method chaining.

<a id="chain_factory.ChainFactory.use_default_token_splitter"></a>

#### use\_default\_token\_splitter

```python
def use_default_token_splitter(chunk_size: int = 500,
                               chunk_overlap: int = 50) -> Self
```

Sets up a Token TextSplitter with the provided values or the default ones if omitted

<a id="chain_factory.ChainFactory.use_default_text_splitter"></a>

#### use\_default\_text\_splitter

```python
def use_default_text_splitter(chunk_size: int = 1000,
                              chunk_overlap: int = 200) -> Self
```

Sets up a Recursive TextSplitter with the provided values or the default ones if omitted

<a id="chain_factory.ChainFactory.use_rag"></a>

#### use\_rag

```python
def use_rag(rag: bool) -> Self
```

Enables or disables the use of Retrieval-Augmented Generation (RAG) in the chain.

**Arguments**:

- `rag`: A boolean flag to enable or disable RAG.

**Returns**:

The ChainFactory instance for method chaining.

<a id="chain_factory.ChainFactory.use_rag_returning_sources"></a>

#### use\_rag\_returning\_sources

```python
def use_rag_returning_sources(returning_sources: bool,
                              format_as_string: bool = False) -> Self
```

Sets whether the RAG component should return source documents along with the generated answer.

**Arguments**:

- `returning_sources`: A boolean flag to indicate if sources should be returned.

**Returns**:

The ChainFactory instance for method chaining.

<a id="chain_factory.ChainFactory.use_embeddings"></a>

#### use\_embeddings

```python
def use_embeddings(embeddings: Embeddings) -> Self
```

Sets the embeddings instance.

**Arguments**:

- `embeddings`: An instance of Embeddings to set.

**Returns**:

The ChainFactory instance for method chaining.

<a id="chain_factory.ChainFactory.use_vector_store"></a>

#### use\_vector\_store

```python
def use_vector_store(vector_store: VectorStore) -> Self
```

Sets the vector store instance and enables Retrieval-Augmented Generation (RAG).

By default, the vector store is also used as a retriever.

**Arguments**:

- `vector_store`: An instance of VectorStore to set.

**Returns**:

The ChainFactory instance for method chaining.

<a id="chain_factory.ChainFactory.use_retriever"></a>

#### use\_retriever

```python
@overload
def use_retriever(retriever: RetrieverLike) -> Self
```

Sets the retriever instance.

**Arguments**:

- `retriever`: An instance of RetrieverLike to set.

**Returns**:

The ChainFactory instance for method chaining.

<a id="chain_factory.ChainFactory.use_retriever"></a>

#### use\_retriever

```python
@overload
def use_retriever(
    retriever: Callable[[LanguageModelLike, RetrieverLike | None],
                        RetrieverLike]
) -> Self
```

Sets the retriever instance using a callable builder.

**Arguments**:

- `retriever`: A callable that takes a LanguageModelLike instance and an optional existing retriever
to produce a new RetrieverLike instance.

**Returns**:

The ChainFactory instance for method chaining.

<a id="chain_factory.ChainFactory.use_retriever"></a>

#### use\_retriever

```python
def use_retriever(
    retriever: RetrieverLike
    | Callable[[LanguageModelLike, RetrieverLike | None], RetrieverLike]
    | None = None
) -> Self
```

Sets a custom retriever instance or builds one using the provided callable.

This method ensures retrieval-augmented generation (RAG) is enabled and assigns the retriever
provided. If the retriever is given as a callable, it evaluates the callable with the current
language model and the existing retriever (if any) to construct a new retriever.

**Arguments**:

- `retriever`: Either a `RetrieverLike` instance or a callable that takes a `LanguageModelLike`
instance and an optional existing retriever to produce a new one.

**Returns**:

The ChainFactory instance for method chaining.

<a id="chain_factory.ChainFactory.ingestor"></a>

#### ingestor

```python
@property
def ingestor() -> RagDocumentIngestor
```

Creates and returns an instance of RagDocumentIngestor.

This method initializes a RagDocumentIngestor using the currently set vector
store and text splitter. These components must be configured
prior to calling this method, otherwise, an error will be raised.

**Raises**:

- `RuntimeError`: If either vector store or text splitter is not set.

**Returns**:

A configured RagDocumentIngestor instance.

<a id="chain_factory.ChainFactory.build"></a>

#### build

```python
def build() -> Runnable[ChainInputType, ChainOutputVar]
```

Constructs and returns the complete runnable chain, either with or without

Retrieval-Augmented Generation (RAG) components based on the current configuration.

If RAG is enabled (`use_rag`), the chain handles retrieval and integration
of external context documents into the generation process. If `rag_return_sources`
is set, it ensures source documents are included in the output.

**Returns**:

A RunnableSerializable instance representing the complete chain.

