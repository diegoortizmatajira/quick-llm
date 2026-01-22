<a id="rag_document_ingestor"></a>

# rag\_document\_ingestor

RAG Document Ingestor Module

<a id="rag_document_ingestor.RagDocumentIngestor"></a>

## RagDocumentIngestor Objects

```python
class RagDocumentIngestor()
```

A class responsible for ingesting documents into a vector database system.

This class allows the ingestion of documents sourced from various formats
such as Markdown, Text, CSV, HTML, JSON, and PDF. Users can either use
the provided loaders or pass their own documents to be processed. Optionally,
a text splitter can be used to preprocess the documents before storage.
All ingested data is persisted directly into the vector database.

<a id="rag_document_ingestor.RagDocumentIngestor.from_loader"></a>

#### from\_loader

```python
def from_loader(loader: BaseLoader, *, use_splitter: bool = True)
```

Ingests documents into the vector database using the specified document loader.

This method utilizes a document loader to fetch and load documents, which
are then ingested into the vector database either directly or after being
processed by a text splitter, if specified.

**Arguments**:

- `loader` _BaseLoader_ - The document loader used to load documents from a source.
- `use_splitter` _bool, optional_ - A flag indicating whether to preprocess
  the documents using the text splitter before ingesting. Defaults to True.
  

**Returns**:

- `Self` - Returns the instance of the RagDocumentIngestor for method chaining.

<a id="rag_document_ingestor.RagDocumentIngestor.from_documents"></a>

#### from\_documents

```python
def from_documents(docs: list[Document], *, use_splitter: bool = True)
```

Ingests a list of documents into the vector database.

This method manages the ingestion process of documents by either
directly adding them to the vector database or preprocessing them
using a text splitter, if specified.

**Arguments**:

- `docs` _list[Document]_ - The list of documents to be ingested.
- `use_splitter` _bool, optional_ - A flag indicating whether to preprocess
  the documents using the text splitter before ingesting. Defaults to True.
  

**Returns**:

- `Self` - Returns the instance of the RagDocumentIngestor for method chaining.
  

**Raises**:

- `RuntimeError` - If the vector store is not set before ingesting documents.
- `RuntimeError` - If the text splitter is not set while `use_splitter` is True.

<a id="rag_document_ingestor.RagDocumentIngestor.from_markdown_document"></a>

#### from\_markdown\_document

```python
def from_markdown_document(source_file: str,
                           *,
                           use_splitter: bool = True,
                           **kwargs) -> Self
```

Ingests a Markdown document into the vector database.

This method loads a Markdown document from the specified source file and processes
it for storage in the vector database. The document can optionally be preprocessed
using a text splitter before ingestion.

**Arguments**:

- `source_file` _str_ - The path to the Markdown document to be ingested.
- `use_splitter` _bool, optional_ - A flag indicating whether to preprocess
  the document using the text splitter before ingestion. Defaults to True.
- `**kwargs` - Additional keyword arguments passed to the loader.
  

**Returns**:

- `Self` - Returns the instance of the RagDocumentIngestor for method chaining.

<a id="rag_document_ingestor.RagDocumentIngestor.from_text_document"></a>

#### from\_text\_document

```python
def from_text_document(source_file: str,
                       *,
                       use_splitter: bool = True,
                       **kwargs) -> Self
```

Ingests a plain text document into the vector database.

This method loads a text document from the specified source file and processes
it for storage in the vector database. The document can optionally be preprocessed
using a text splitter before ingestion.

**Arguments**:

- `source_file` _str_ - The path to the text document to be ingested.
- `use_splitter` _bool, optional_ - A flag indicating whether to preprocess
  the document using the text splitter before ingestion. Defaults to True.
- `**kwargs` - Additional keyword arguments passed to the loader.
  

**Returns**:

- `Self` - Returns the instance of the RagDocumentIngestor for method chaining.

<a id="rag_document_ingestor.RagDocumentIngestor.from_csv_file"></a>

#### from\_csv\_file

```python
def from_csv_file(source_file: str,
                  *,
                  use_splitter: bool = True,
                  **kwargs) -> Self
```

Ingests a CSV document into the vector database.

This method loads a CSV document from the specified source file and processes
it for storage in the vector database. The document can optionally be preprocessed
using a text splitter before ingestion.

**Arguments**:

- `source_file` _str_ - The path to the CSV document to be ingested.
- `use_splitter` _bool, optional_ - A flag indicating whether to preprocess
  the document using the text splitter before ingestion. Defaults to True.
- `**kwargs` - Additional keyword arguments passed to the loader.
  

**Returns**:

- `Self` - Returns the instance of the RagDocumentIngestor for method chaining.

<a id="rag_document_ingestor.RagDocumentIngestor.from_documents_folder"></a>

#### from\_documents\_folder

```python
def from_documents_folder(path: str,
                          glob: str,
                          *,
                          use_splitter: bool = True,
                          **kwargs) -> Self
```

Ingests a folder of documents into the vector database.

This method loads multiple documents from the specified folder and processes
them for storage in the vector database. Optionally, a glob pattern can be
used to specify which files to load, and the documents can be preprocessed
with a text splitter before ingestion.

**Arguments**:

- `path` _str_ - The path to the folder containing the documents to be ingested.
- `glob` _str_ - The glob pattern to match files within the folder.
- `use_splitter` _bool, optional_ - A flag indicating whether to preprocess
  the documents using the text splitter before ingestion. Defaults to True.
- `**kwargs` - Additional keyword arguments passed to the loader.
  

**Returns**:

- `Self` - Returns the instance of the RagDocumentIngestor for method chaining.

<a id="rag_document_ingestor.RagDocumentIngestor.from_html_document"></a>

#### from\_html\_document

```python
def from_html_document(source_file: str,
                       *,
                       use_splitter: bool = True,
                       **kwargs) -> Self
```

Ingests an HTML document into the vector database.

This method loads an HTML document from the specified source file and processes
it for storage in the vector database. The document can optionally be preprocessed
using a text splitter before ingestion.

**Arguments**:

- `source_file` _str_ - The path to the HTML document to be ingested.
- `use_splitter` _bool, optional_ - A flag indicating whether to preprocess
  the document using the text splitter before ingestion. Defaults to True.
- `**kwargs` - Additional keyword arguments passed to the loader.
  

**Returns**:

- `Self` - Returns the instance of the RagDocumentIngestor for method chaining.

<a id="rag_document_ingestor.RagDocumentIngestor.from_html_document_with_beautifulsoup"></a>

#### from\_html\_document\_with\_beautifulsoup

```python
def from_html_document_with_beautifulsoup(source_file: str,
                                          *,
                                          use_splitter: bool = True,
                                          **kwargs) -> Self
```

Ingests an HTML document into the vector database using BeautifulSoup.

This method loads an HTML document from the specified source file using
the BSHTMLLoader and processes it for storage in the vector database.
The document can optionally be preprocessed using a text splitter
before ingestion.

**Arguments**:

- `source_file` _str_ - The path to the HTML document to be ingested.
- `use_splitter` _bool, optional_ - A flag indicating whether to preprocess
  the document using the text splitter before ingestion. Defaults to True.
- `**kwargs` - Additional keyword arguments passed to the loader.
  

**Returns**:

- `Self` - Returns the instance of the RagDocumentIngestor for method chaining.

<a id="rag_document_ingestor.RagDocumentIngestor.from_json_document"></a>

#### from\_json\_document

```python
def from_json_document(source_file: str,
                       *,
                       use_splitter: bool = True,
                       **kwargs) -> Self
```

Ingests a JSON document into the vector database.

This method loads a JSON document from the specified source file and processes
it for storage in the vector database. The document can optionally be preprocessed
using a text splitter before ingestion.

**Arguments**:

- `source_file` _str_ - The path to the JSON document to be ingested.
- `use_splitter` _bool, optional_ - A flag indicating whether to preprocess
  the document using the text splitter before ingestion. Defaults to True.
- `**kwargs` - Additional keyword arguments passed to the loader.
  

**Returns**:

- `Self` - Returns the instance of the RagDocumentIngestor for method chaining.

<a id="rag_document_ingestor.RagDocumentIngestor.from_pdf_document"></a>

#### from\_pdf\_document

```python
def from_pdf_document(source_file: str,
                      *,
                      use_splitter: bool = True,
                      **kwargs) -> Self
```

Ingests a PDF document into the vector database.

This method loads a PDF document from the specified source file and processes
it for storage in the vector database. The document can optionally be preprocessed
using a text splitter before ingestion.

**Arguments**:

- `source_file` _str_ - The path to the PDF document to be ingested.
- `use_splitter` _bool, optional_ - A flag indicating whether to preprocess
  the document using the text splitter before ingestion. Defaults to True.
- `**kwargs` - Additional keyword arguments passed to the loader.
  

**Returns**:

- `Self` - Returns the instance of the RagDocumentIngestor for method chaining.

