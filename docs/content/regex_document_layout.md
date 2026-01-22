<a id="preprocessor.regex_document_layout"></a>

# preprocessor.regex\_document\_layout

This module defines the class that defines the rules to process a given document layout

<a id="preprocessor.regex_document_layout.RegexDocumentLayout"></a>

## RegexDocumentLayout Objects

```python
class RegexDocumentLayout()
```

A class that defines the rules to process a given document layout.

This class provides functionality for selecting, processing, and cleaning documents using
regular expressions. It also facilitates skipping documents based on defined conditions and
retrieving document layouts that match specific selectors.

**Attributes**:

- `selector_pattern` _str_ - Regular expression for checking if an instance matches a selector value.
- `line_cleaning_regex` _Optional[list[str]]_ - Regular expressions for removing lines where a match is found.
- `global_cleaning_regex` _Optional[list[str]]_ - Regular expressions for removing multi-line content.
- `skip_doc_regex` _Optional[str]_ - Regular expression for skipping certain documents.
- `remove_empty_lines` _bool_ - Whether to remove empty lines during preprocessing.
- `remove_empty_docs` _bool_ - Whether to remove empty documents during preprocessing.
- `remove_leading_spaces` _bool_ - Whether to remove leading spaces in lines during preprocessing.
- `remove_trailing_spaces` _bool_ - Whether to remove trailing spaces in lines during preprocessing.

<a id="preprocessor.regex_document_layout.RegexDocumentLayout.__init__"></a>

#### \_\_init\_\_

```python
def __init__(*,
             selector_pattern: str = r".*",
             line_cleaning_regex: Optional[list[str]] = None,
             global_cleaning_regex: Optional[list[str]] = None,
             skip_doc_regex: Optional[str] = None,
             remove_empty_lines: bool = True,
             remove_empty_docs: bool = True,
             remove_leading_spaces_in_lines: bool = True,
             remove_trailing_spaces_in_lines: bool = True)
```

Initializes a RegexDocumentLayout instance with the provided configuration.

**Arguments**:

- `selector_pattern` _str_ - Regular expression used to check if an instance matches a selector.
- `line_cleaning_regex` _Optional[list[str]]_ - List of regex patterns for cleaning lines that match.
- `global_cleaning_regex` _Optional[list[str]]_ - List of regex patterns for cleaning multi-line content.
- `skip_doc_regex` _Optional[str]_ - Regex pattern that determines if a document should be skipped.
- `remove_empty_lines` _bool_ - If True, removes empty lines during preprocessing.
- `remove_empty_docs` _bool_ - If True, skips documents that are empty.
- `remove_leading_spaces_in_lines` _bool_ - If True, removes leading spaces from lines during preprocessing.
- `remove_trailing_spaces_in_lines` _bool_ - If True, removes trailing spaces from lines during preprocessing.

<a id="preprocessor.regex_document_layout.RegexDocumentLayout.matches_selector"></a>

#### matches\_selector

```python
def matches_selector(selector: str) -> bool
```

Determines if the given selector matches the layout's selector pattern.

**Arguments**:

- `selector` _str_ - The selector to check against the layout's pattern.
  

**Returns**:

- `bool` - True if the selector matches the pattern, False otherwise.

<a id="preprocessor.regex_document_layout.RegexDocumentLayout.should_skip_doc"></a>

#### should\_skip\_doc

```python
def should_skip_doc(doc_content: str) -> bool
```

Determines whether a document should be skipped based on its content.

**Arguments**:

- `doc_content` _str_ - The content of the document to evaluate.
  

**Returns**:

- `bool` - True if the document should be skipped, False otherwise.

<a id="preprocessor.regex_document_layout.RegexDocumentLayout.clean_content"></a>

#### clean\_content

```python
def clean_content(doc_content: str) -> str
```

Cleans the content of a document based on the defined cleaning rules.

This method applies both global and line-specific cleaning regular expressions to
modify the content of the document. Global cleaning is applied over the entire
document, and line-specific cleaning processes each line separately. Processed lines
are then reassembled into the final cleaned document.

**Arguments**:

- `doc_content` _str_ - The original content of the document.
  

**Returns**:

- `str` - The cleaned content of the document.

<a id="preprocessor.regex_document_layout.RegexDocumentLayout.skip_or_process_document"></a>

#### skip\_or\_process\_document

```python
def skip_or_process_document(doc_content: str) -> Tuple[bool, str]
```

Determines whether to skip or process a given document and cleans its content.

This method evaluates the document's content to decide if it should be skipped
based on the defined conditions. If the document is not skipped, its content is
cleaned according to the rules specified for the layout, such as removing empty
lines, leading/trailing spaces, and applying regular expressions for line/global
cleaning.

**Arguments**:

- `doc_content` _str_ - The content of the document to evaluate and process.
  

**Returns**:

  Tuple[bool, str]: A tuple where the first element is a boolean indicating if
  the document should be skipped, and the second element is
  the cleaned document content (or an empty string if skipped).

<a id="preprocessor.regex_document_layout.RegexDocumentLayout.get_matching_layout"></a>

#### get\_matching\_layout

```python
@staticmethod
def get_matching_layout(layout_collection: list["RegexDocumentLayout"],
                        selector: str) -> Optional["RegexDocumentLayout"]
```

Retrieves the first layout that matches the given selector from a collection of layouts.

This method iterates over a collection of RegexDocumentLayout instances and
returns the first one whose selector pattern matches the provided selector.

**Arguments**:

- `layout_collection` _list["RegexDocumentLayout"]_ - A list of RegexDocumentLayout instances to evaluate.
- `selector` _str_ - The selector string to match against the layouts' selector patterns.
  

**Returns**:

- `Optional["RegexDocumentLayout"]` - The first RegexDocumentLayout that matches the selector,
  or None if no match is found.

