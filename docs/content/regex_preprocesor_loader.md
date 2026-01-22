<a id="preprocessor.regex_preprocesor_loader"></a>

# preprocessor.regex\_preprocesor\_loader

This module defines a Preprocessor middleware that can be used along with any
other loader to executing a preprocessing cleanup based on regular expressions

<a id="preprocessor.regex_preprocesor_loader.RegexPreprocessorLoader"></a>

## RegexPreprocessorLoader Objects

```python
class RegexPreprocessorLoader(BaseLoader)
```

A document loader middleware that preprocesses documents using customizable
regular expression-based layouts. This extends the functionality of a source
document loader by applying preprocessing rules defined in the layouts.

The preprocessing involves transforming the content of the documents or skipping
some documents entirely based on the rules defined in the provided layouts.

**Attributes**:

- `source_loader` _BaseLoader_ - The original document loader whose output will
  be preprocessed.
- `layout_list` _list[RegexDocumentLayout]_ - A collection of layouts specifying
  regex patterns and preprocessing rules.
- `layout_selector` _Callable[[Document], str]_ - A function that determines which
  layout from the layout_list should be used for a given document.

<a id="preprocessor.regex_preprocesor_loader.RegexPreprocessorLoader.__init__"></a>

#### \_\_init\_\_

```python
def __init__(source_loader: BaseLoader, layout_list: list[RegexDocumentLayout],
             layout_selector: Callable[[Document], str])
```

Initializes the RegexPreprocessorLoader.

**Arguments**:

- `source_loader` _BaseLoader_ - The original document loader whose output will be processed.
- `layout_list` _list[RegexDocumentLayout]_ - A list of RegexDocumentLayout objects used to
  define the preprocessing patterns and rules.
- `layout_selector` _Callable[[Document], str]_ - A callable that takes a Document and returns
  a selector value used to match the appropriate layout from the layout_list.

<a id="preprocessor.regex_preprocesor_loader.RegexPreprocessorLoader.lazy_load"></a>

#### lazy\_load

```python
def lazy_load() -> Iterator[Document]
```

Processes documents from the source loader, applies the regex preprocessing
based on matching layouts, and yields the processed documents.

- For each document obtained from the source loader, the `layout_selector`
is used to determine the appropriate layout from the `layout_list`.
- If a matching layout is found, its `skip_or_process_document` method
is invoked to preprocess the document content and decide whether the
document should be skipped.
- Only non-skipped documents are yielded.

**Returns**:

- `Iterator[Document]` - An iterator over the processed documents.

