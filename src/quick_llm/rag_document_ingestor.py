"""RAG Document Ingestor Module"""

from typing import Self
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_community.document_loaders import (
    BSHTMLLoader,
    CSVLoader,
    DirectoryLoader,
    JSONLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.document_loaders.base import BaseLoader
from langchain_text_splitters import TextSplitter


class RagDocumentIngestor:
    """
    A class responsible for ingesting documents into a vector database system.

    This class allows the ingestion of documents sourced from various formats
    such as Markdown, Text, CSV, HTML, JSON, and PDF. Users can either use
    the provided loaders or pass their own documents to be processed. Optionally,
    a text splitter can be used to preprocess the documents before storage.
    All ingested data is persisted directly into the vector database.
    """

    def __init__(
        self, vector_store: VectorStore, text_splitter: TextSplitter | None = None
    ):
        self.__vector_store = vector_store
        self.__text_splitter = text_splitter

    def from_loader(self, loader: BaseLoader, *, use_splitter: bool = True):
        """
        Ingests documents into the vector database using the specified document loader.

        This method utilizes a document loader to fetch and load documents, which
        are then ingested into the vector database either directly or after being
        processed by a text splitter, if specified.

        Args:
            loader (BaseLoader): The document loader used to load documents from a source.
            use_splitter (bool, optional): A flag indicating whether to preprocess
                the documents using the text splitter before ingesting. Defaults to True.

        Returns:
            Self: Returns the instance of the RagDocumentIngestor for method chaining.
        """
        docs = loader.load()
        return self.from_documents(docs, use_splitter=use_splitter)

    def from_documents(self, docs: list[Document], *, use_splitter: bool = True):
        """
        Ingests a list of documents into the vector database.

        This method manages the ingestion process of documents by either
        directly adding them to the vector database or preprocessing them
        using a text splitter, if specified.

        Args:
            docs (list[Document]): The list of documents to be ingested.
            use_splitter (bool, optional): A flag indicating whether to preprocess
                the documents using the text splitter before ingesting. Defaults to True.

        Returns:
            Self: Returns the instance of the RagDocumentIngestor for method chaining.

        Raises:
            RuntimeError: If the vector store is not set before ingesting documents.
            RuntimeError: If the text splitter is not set while `use_splitter` is True.
        """
        if self.__vector_store is None:
            raise RuntimeError(
                "You must select an vector Db before ingesting documents."
            )
        if use_splitter:
            if not self.__text_splitter:
                raise RuntimeError(
                    "You must select a text splitter before ingesting documents."
                )
            splits = self.__text_splitter.split_documents(docs)
            self.__vector_store.add_documents(splits)
        else:
            self.__vector_store.add_documents(docs)
        return self

    def from_markdown_document(
        self, source_file: str, *, use_splitter: bool = True, **kwargs
    ) -> Self:
        """
        Ingests a Markdown document into the vector database.

        This method loads a Markdown document from the specified source file and processes
        it for storage in the vector database. The document can optionally be preprocessed
        using a text splitter before ingestion.

        Args:
            source_file (str): The path to the Markdown document to be ingested.
            use_splitter (bool, optional): A flag indicating whether to preprocess
                the document using the text splitter before ingestion. Defaults to True.
            **kwargs: Additional keyword arguments passed to the loader.

        Returns:
            Self: Returns the instance of the RagDocumentIngestor for method chaining.
        """
        loader = UnstructuredMarkdownLoader(source_file, **kwargs)
        return self.from_loader(loader, use_splitter=use_splitter)

    def from_text_document(
        self, source_file: str, *, use_splitter: bool = True, **kwargs
    ) -> Self:
        """
        Ingests a plain text document into the vector database.

        This method loads a text document from the specified source file and processes
        it for storage in the vector database. The document can optionally be preprocessed
        using a text splitter before ingestion.

        Args:
            source_file (str): The path to the text document to be ingested.
            use_splitter (bool, optional): A flag indicating whether to preprocess
                the document using the text splitter before ingestion. Defaults to True.
            **kwargs: Additional keyword arguments passed to the loader.

        Returns:
            Self: Returns the instance of the RagDocumentIngestor for method chaining.
        """
        loader = TextLoader(source_file, **kwargs)
        return self.from_loader(loader, use_splitter=use_splitter)

    def from_csv_file(
        self, source_file: str, *, use_splitter: bool = True, **kwargs
    ) -> Self:
        """
        Ingests a CSV document into the vector database.

        This method loads a CSV document from the specified source file and processes
        it for storage in the vector database. The document can optionally be preprocessed
        using a text splitter before ingestion.

        Args:
            source_file (str): The path to the CSV document to be ingested.
            use_splitter (bool, optional): A flag indicating whether to preprocess
                the document using the text splitter before ingestion. Defaults to True.
            **kwargs: Additional keyword arguments passed to the loader.

        Returns:
            Self: Returns the instance of the RagDocumentIngestor for method chaining.
        """
        loader = CSVLoader(source_file, **kwargs)
        return self.from_loader(loader, use_splitter=use_splitter)

    def from_documents_folder(
        self, path: str, glob: str, *, use_splitter: bool = True, **kwargs
    ) -> Self:
        """
        Ingests a folder of documents into the vector database.

        This method loads multiple documents from the specified folder and processes
        them for storage in the vector database. Optionally, a glob pattern can be
        used to specify which files to load, and the documents can be preprocessed
        with a text splitter before ingestion.

        Args:
            path (str): The path to the folder containing the documents to be ingested.
            glob (str): The glob pattern to match files within the folder.
            use_splitter (bool, optional): A flag indicating whether to preprocess
                the documents using the text splitter before ingestion. Defaults to True.
            **kwargs: Additional keyword arguments passed to the loader.

        Returns:
            Self: Returns the instance of the RagDocumentIngestor for method chaining.
        """
        loader = DirectoryLoader(path, glob, **kwargs)
        return self.from_loader(loader, use_splitter=use_splitter)

    def from_html_document(
        self, source_file: str, *, use_splitter: bool = True, **kwargs
    ) -> Self:
        """
        Ingests an HTML document into the vector database.

        This method loads an HTML document from the specified source file and processes
        it for storage in the vector database. The document can optionally be preprocessed
        using a text splitter before ingestion.

        Args:
            source_file (str): The path to the HTML document to be ingested.
            use_splitter (bool, optional): A flag indicating whether to preprocess
                the document using the text splitter before ingestion. Defaults to True.
            **kwargs: Additional keyword arguments passed to the loader.

        Returns:
            Self: Returns the instance of the RagDocumentIngestor for method chaining.
        """
        loader = UnstructuredHTMLLoader(source_file, **kwargs)
        return self.from_loader(loader, use_splitter=use_splitter)

    def from_html_document_with_beautifulsoup(
        self, source_file: str, *, use_splitter: bool = True, **kwargs
    ) -> Self:
        """
        Ingests an HTML document into the vector database using BeautifulSoup.

        This method loads an HTML document from the specified source file using
        the BSHTMLLoader and processes it for storage in the vector database.
        The document can optionally be preprocessed using a text splitter
        before ingestion.

        Args:
            source_file (str): The path to the HTML document to be ingested.
            use_splitter (bool, optional): A flag indicating whether to preprocess
                the document using the text splitter before ingestion. Defaults to True.
            **kwargs: Additional keyword arguments passed to the loader.

        Returns:
            Self: Returns the instance of the RagDocumentIngestor for method chaining.
        """
        loader = BSHTMLLoader(source_file, **kwargs)
        return self.from_loader(loader, use_splitter=use_splitter)

    def from_json_document(
        self, source_file: str, *, use_splitter: bool = True, **kwargs
    ) -> Self:
        """
        Ingests a JSON document into the vector database.

        This method loads a JSON document from the specified source file and processes
        it for storage in the vector database. The document can optionally be preprocessed
        using a text splitter before ingestion.

        Args:
            source_file (str): The path to the JSON document to be ingested.
            use_splitter (bool, optional): A flag indicating whether to preprocess
                the document using the text splitter before ingestion. Defaults to True.
            **kwargs: Additional keyword arguments passed to the loader.

        Returns:
            Self: Returns the instance of the RagDocumentIngestor for method chaining.
        """
        loader = JSONLoader(source_file, **kwargs)
        return self.from_loader(loader, use_splitter=use_splitter)

    def from_pdf_document(
        self, source_file: str, *, use_splitter: bool = True, **kwargs
    ) -> Self:
        """
        Ingests a PDF document into the vector database.

        This method loads a PDF document from the specified source file and processes
        it for storage in the vector database. The document can optionally be preprocessed
        using a text splitter before ingestion.

        Args:
            source_file (str): The path to the PDF document to be ingested.
            use_splitter (bool, optional): A flag indicating whether to preprocess
                the document using the text splitter before ingestion. Defaults to True.
            **kwargs: Additional keyword arguments passed to the loader.

        Returns:
            Self: Returns the instance of the RagDocumentIngestor for method chaining.
        """
        loader = PyPDFLoader(source_file, **kwargs)
        return self.from_loader(loader, use_splitter=use_splitter)
