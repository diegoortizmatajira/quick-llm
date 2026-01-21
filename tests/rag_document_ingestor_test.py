"""Tests for the RagDocumentIngestor class"""

from unittest.mock import Mock, patch

import pytest
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

from quick_llm import RagDocumentIngestor

# Test data constants
TEST_DOCUMENT1 = Document(
    page_content="This is the first test document with some content.",
    metadata={"source": "test1.txt"},
)
TEST_DOCUMENT2 = Document(
    page_content="This is the second test document with different content.",
    metadata={"source": "test2.txt"},
)
TEST_DOCUMENT3 = Document(
    page_content="A third document for testing multiple document ingestion.",
    metadata={"source": "test3.txt"},
)
TEST_DOCUMENTS = [TEST_DOCUMENT1, TEST_DOCUMENT2, TEST_DOCUMENT3]

TEST_SPLIT_DOC1 = Document(page_content="This is the first", metadata={"source": "test1.txt"})
TEST_SPLIT_DOC2 = Document(page_content="test document", metadata={"source": "test1.txt"})
TEST_SPLIT_DOCS = [TEST_SPLIT_DOC1, TEST_SPLIT_DOC2]


def _create_vector_store() -> InMemoryVectorStore:
    """Create a test vector store with fake embeddings"""
    return InMemoryVectorStore(FakeEmbeddings(size=10))


def _create_text_splitter() -> CharacterTextSplitter:
    """Create a test text splitter"""
    return CharacterTextSplitter(chunk_size=20, chunk_overlap=0)


def _create_mock_loader(documents: list[Document]) -> Mock:
    """Create a mock loader that returns specified documents"""
    mock_loader = Mock(spec=BaseLoader)
    mock_loader.load.return_value = documents
    return mock_loader


class TestRagDocumentIngestorInitialization:
    """Test RagDocumentIngestor initialization"""

    def test_init_with_vector_store_only(self):
        """Test initialization with only vector store"""
        vector_store = _create_vector_store()
        ingestor = RagDocumentIngestor(vector_store)
        assert ingestor is not None

    def test_init_with_vector_store_and_splitter(self):
        """Test initialization with vector store and text splitter"""
        vector_store = _create_vector_store()
        text_splitter = _create_text_splitter()
        ingestor = RagDocumentIngestor(vector_store, text_splitter)
        assert ingestor is not None


class TestFromDocuments:
    """Test from_documents method"""

    def test_from_documents_without_splitter(self):
        """Test ingesting documents without using text splitter"""
        vector_store = _create_vector_store()
        ingestor = RagDocumentIngestor(vector_store)

        result = ingestor.from_documents(TEST_DOCUMENTS, use_splitter=False)

        # Verify method chaining
        assert result is ingestor

        # Verify documents were added to vector store
        search_results = vector_store.similarity_search("test document", k=5)
        assert len(search_results) >= 3

    def test_from_documents_with_splitter(self):
        """Test ingesting documents with text splitter"""
        vector_store = _create_vector_store()
        text_splitter = _create_text_splitter()
        ingestor = RagDocumentIngestor(vector_store, text_splitter)

        result = ingestor.from_documents(TEST_DOCUMENTS, use_splitter=True)

        # Verify method chaining
        assert result is ingestor

        # Verify documents were added (may or may not be split depending on content length)
        search_results = vector_store.similarity_search("test", k=10)
        assert len(search_results) >= len(TEST_DOCUMENTS)

    def test_from_documents_without_vector_store(self):
        """Test that RuntimeError is raised when vector store is None"""
        ingestor = RagDocumentIngestor(None)  # pyright: ignore[reportArgumentType]

        with pytest.raises(RuntimeError, match="You must select an vector Db"):
            ingestor.from_documents(TEST_DOCUMENTS, use_splitter=False)

    def test_from_documents_use_splitter_without_text_splitter(self):
        """Test that RuntimeError is raised when use_splitter=True but no text splitter"""
        vector_store = _create_vector_store()
        ingestor = RagDocumentIngestor(vector_store, text_splitter=None)

        with pytest.raises(RuntimeError, match="You must select a text splitter"):
            ingestor.from_documents(TEST_DOCUMENTS, use_splitter=True)

    def test_from_documents_empty_list(self):
        """Test ingesting an empty list of documents"""
        vector_store = _create_vector_store()
        ingestor = RagDocumentIngestor(vector_store)

        result = ingestor.from_documents([], use_splitter=False)

        assert result is ingestor


class TestFromLoader:
    """Test from_loader method"""

    def test_from_loader_with_splitter(self):
        """Test ingesting documents from loader with text splitter"""
        vector_store = _create_vector_store()
        text_splitter = _create_text_splitter()
        ingestor = RagDocumentIngestor(vector_store, text_splitter)

        mock_loader = _create_mock_loader(TEST_DOCUMENTS)
        result = ingestor.from_loader(mock_loader, use_splitter=True)

        # Verify loader was called
        mock_loader.load.assert_called_once()

        # Verify method chaining
        assert result is ingestor

        # Verify documents were added
        search_results = vector_store.similarity_search("test", k=10)
        assert len(search_results) > 0

    def test_from_loader_without_splitter(self):
        """Test ingesting documents from loader without text splitter"""
        vector_store = _create_vector_store()
        ingestor = RagDocumentIngestor(vector_store)

        mock_loader = _create_mock_loader(TEST_DOCUMENTS)
        result = ingestor.from_loader(mock_loader, use_splitter=False)

        # Verify loader was called
        mock_loader.load.assert_called_once()

        # Verify method chaining
        assert result is ingestor

        # Verify documents were added
        search_results = vector_store.similarity_search("test document", k=5)
        assert len(search_results) >= 3

    def test_from_loader_empty_results(self):
        """Test from_loader when loader returns empty list"""
        vector_store = _create_vector_store()
        ingestor = RagDocumentIngestor(vector_store)

        mock_loader = _create_mock_loader([])
        result = ingestor.from_loader(mock_loader, use_splitter=False)

        mock_loader.load.assert_called_once()
        assert result is ingestor


class TestMethodChaining:
    """Test that multiple ingestion operations can be chained"""

    def test_chained_from_documents_calls(self):
        """Test chaining multiple from_documents calls"""
        vector_store = _create_vector_store()
        ingestor = RagDocumentIngestor(vector_store)

        result = (
            ingestor
            .from_documents([TEST_DOCUMENT1], use_splitter=False)
            .from_documents([TEST_DOCUMENT2], use_splitter=False)
            .from_documents([TEST_DOCUMENT3], use_splitter=False)
        )

        assert result is ingestor

        # Verify all documents were added
        search_results = vector_store.similarity_search("document", k=10)
        assert len(search_results) >= 3

    def test_chained_from_loader_calls(self):
        """Test chaining multiple from_loader calls"""
        vector_store = _create_vector_store()
        ingestor = RagDocumentIngestor(vector_store)

        mock_loader1 = _create_mock_loader([TEST_DOCUMENT1])
        mock_loader2 = _create_mock_loader([TEST_DOCUMENT2])

        result = (
            ingestor
            .from_loader(mock_loader1, use_splitter=False)
            .from_loader(mock_loader2, use_splitter=False)
        )

        assert result is ingestor
        mock_loader1.load.assert_called_once()
        mock_loader2.load.assert_called_once()


class TestSpecificFormatMethods:
    """Test format-specific ingestion methods"""

    @patch("quick_llm.rag_document_ingestor.UnstructuredMarkdownLoader")
    def test_from_markdown_document(self, mock_loader_class):
        """Test ingesting markdown document"""
        vector_store = _create_vector_store()
        text_splitter = _create_text_splitter()
        ingestor = RagDocumentIngestor(vector_store, text_splitter)

        mock_loader = _create_mock_loader(TEST_DOCUMENTS)
        mock_loader_class.return_value = mock_loader

        result = ingestor.from_markdown_document("test.md", use_splitter=True, mode="single")

        # Verify loader was created with correct arguments
        mock_loader_class.assert_called_once_with("test.md", mode="single")
        mock_loader.load.assert_called_once()
        assert result is ingestor

    @patch("quick_llm.rag_document_ingestor.TextLoader")
    def test_from_text_document(self, mock_loader_class):
        """Test ingesting text document"""
        vector_store = _create_vector_store()
        ingestor = RagDocumentIngestor(vector_store)

        mock_loader = _create_mock_loader(TEST_DOCUMENTS)
        mock_loader_class.return_value = mock_loader

        result = ingestor.from_text_document("test.txt", use_splitter=False, encoding="utf-8")

        # Verify loader was created with correct arguments
        mock_loader_class.assert_called_once_with("test.txt", encoding="utf-8")
        mock_loader.load.assert_called_once()
        assert result is ingestor

    @patch("quick_llm.rag_document_ingestor.CSVLoader")
    def test_from_csv_file(self, mock_loader_class):
        """Test ingesting CSV file"""
        vector_store = _create_vector_store()
        text_splitter = _create_text_splitter()
        ingestor = RagDocumentIngestor(vector_store, text_splitter)

        mock_loader = _create_mock_loader(TEST_DOCUMENTS)
        mock_loader_class.return_value = mock_loader

        result = ingestor.from_csv_file("test.csv", use_splitter=True)

        mock_loader_class.assert_called_once_with("test.csv")
        mock_loader.load.assert_called_once()
        assert result is ingestor

    @patch("quick_llm.rag_document_ingestor.DirectoryLoader")
    def test_from_documents_folder(self, mock_loader_class):
        """Test ingesting documents from folder"""
        vector_store = _create_vector_store()
        ingestor = RagDocumentIngestor(vector_store)

        mock_loader = _create_mock_loader(TEST_DOCUMENTS)
        mock_loader_class.return_value = mock_loader

        result = ingestor.from_documents_folder(
            "/path/to/docs",
            "*.txt",
            use_splitter=False,
            show_progress=True
        )

        mock_loader_class.assert_called_once_with("/path/to/docs", "*.txt", show_progress=True)
        mock_loader.load.assert_called_once()
        assert result is ingestor

    @patch("quick_llm.rag_document_ingestor.UnstructuredHTMLLoader")
    def test_from_html_document(self, mock_loader_class):
        """Test ingesting HTML document"""
        vector_store = _create_vector_store()
        text_splitter = _create_text_splitter()
        ingestor = RagDocumentIngestor(vector_store, text_splitter)

        mock_loader = _create_mock_loader(TEST_DOCUMENTS)
        mock_loader_class.return_value = mock_loader

        result = ingestor.from_html_document("test.html", use_splitter=True)

        mock_loader_class.assert_called_once_with("test.html")
        mock_loader.load.assert_called_once()
        assert result is ingestor

    @patch("quick_llm.rag_document_ingestor.BSHTMLLoader")
    def test_from_html_document_with_beautifulsoup(self, mock_loader_class):
        """Test ingesting HTML document with BeautifulSoup"""
        vector_store = _create_vector_store()
        ingestor = RagDocumentIngestor(vector_store)

        mock_loader = _create_mock_loader(TEST_DOCUMENTS)
        mock_loader_class.return_value = mock_loader

        result = ingestor.from_html_document_with_beautifulsoup(
            "test.html",
            use_splitter=False,
            bs_kwargs={"features": "html.parser"}
        )

        mock_loader_class.assert_called_once_with("test.html", bs_kwargs={"features": "html.parser"})
        mock_loader.load.assert_called_once()
        assert result is ingestor

    @patch("quick_llm.rag_document_ingestor.JSONLoader")
    def test_from_json_document(self, mock_loader_class):
        """Test ingesting JSON document"""
        vector_store = _create_vector_store()
        text_splitter = _create_text_splitter()
        ingestor = RagDocumentIngestor(vector_store, text_splitter)

        mock_loader = _create_mock_loader(TEST_DOCUMENTS)
        mock_loader_class.return_value = mock_loader

        result = ingestor.from_json_document(
            "test.json",
            use_splitter=True,
            jq_schema=".[]",
            text_content=False
        )

        mock_loader_class.assert_called_once_with(
            "test.json",
            jq_schema=".[]",
            text_content=False
        )
        mock_loader.load.assert_called_once()
        assert result is ingestor

    @patch("quick_llm.rag_document_ingestor.PyPDFLoader")
    def test_from_pdf_document(self, mock_loader_class):
        """Test ingesting PDF document"""
        vector_store = _create_vector_store()
        ingestor = RagDocumentIngestor(vector_store)

        mock_loader = _create_mock_loader(TEST_DOCUMENTS)
        mock_loader_class.return_value = mock_loader

        result = ingestor.from_pdf_document("test.pdf", use_splitter=False)

        mock_loader_class.assert_called_once_with("test.pdf")
        mock_loader.load.assert_called_once()
        assert result is ingestor


class TestTextSplitterIntegration:
    """Test integration with different text splitters"""

    def test_with_character_text_splitter(self):
        """Test using CharacterTextSplitter"""
        vector_store = _create_vector_store()
        text_splitter = CharacterTextSplitter(
            chunk_size=50,
            chunk_overlap=10,
            separator=" "
        )
        ingestor = RagDocumentIngestor(vector_store, text_splitter)

        # Create a long document that will be split
        long_doc = Document(
            page_content="This is a very long document. " * 20,
            metadata={"source": "long.txt"}
        )

        result = ingestor.from_documents([long_doc], use_splitter=True)

        assert result is ingestor
        # Verify document was split into multiple chunks
        # The splitter should have split the 600-character document into multiple 50-char chunks
        search_results = vector_store.similarity_search("long document", k=20)
        assert len(search_results) >= 10, f"Expected at least 10 chunks, got {len(search_results)}"

    def test_with_recursive_character_text_splitter(self):
        """Test using RecursiveCharacterTextSplitter"""
        vector_store = _create_vector_store()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=40,
            chunk_overlap=5,
            separators=["\n\n", "\n", " "]
        )
        ingestor = RagDocumentIngestor(vector_store, text_splitter)

        doc = Document(
            page_content="First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
            metadata={"source": "structured.txt"}
        )

        result = ingestor.from_documents([doc], use_splitter=True)

        assert result is ingestor
        search_results = vector_store.similarity_search("paragraph", k=10)
        assert len(search_results) >= 1


class TestVectorStoreInteraction:
    """Test interaction with vector store"""

    def test_documents_are_searchable_after_ingestion(self):
        """Test that ingested documents can be retrieved through similarity search"""
        vector_store = _create_vector_store()
        ingestor = RagDocumentIngestor(vector_store)

        ingestor.from_documents(TEST_DOCUMENTS, use_splitter=False)

        # Search for content - with FakeEmbeddings we can't guarantee semantic similarity
        # but we can verify documents are retrievable
        results = vector_store.similarity_search("test document", k=3)
        assert len(results) == 3

        # Verify the returned documents are from our test set
        page_contents = [doc.page_content for doc in results]
        assert any("first" in content.lower() or "second" in content.lower() or "third" in content.lower()
                   for content in page_contents)

    def test_metadata_preserved_after_ingestion(self):
        """Test that document metadata is preserved after ingestion"""
        vector_store = _create_vector_store()
        ingestor = RagDocumentIngestor(vector_store)

        doc_with_metadata = Document(
            page_content="Document with metadata",
            metadata={"source": "meta.txt", "author": "Test Author", "date": "2024-01-01"}
        )

        ingestor.from_documents([doc_with_metadata], use_splitter=False)

        results = vector_store.similarity_search("metadata", k=1)
        assert len(results) > 0
        assert results[0].metadata["source"] == "meta.txt"
        assert results[0].metadata["author"] == "Test Author"
        assert results[0].metadata["date"] == "2024-01-01"

    def test_multiple_ingestions_accumulate(self):
        """Test that multiple ingestion operations accumulate documents in vector store"""
        vector_store = _create_vector_store()
        ingestor = RagDocumentIngestor(vector_store)

        # First ingestion
        ingestor.from_documents([TEST_DOCUMENT1], use_splitter=False)
        results_after_first = vector_store.similarity_search("document", k=10)
        count_after_first = len(results_after_first)

        # Second ingestion
        ingestor.from_documents([TEST_DOCUMENT2, TEST_DOCUMENT3], use_splitter=False)
        results_after_second = vector_store.similarity_search("document", k=10)
        count_after_second = len(results_after_second)

        # Verify documents accumulated
        assert count_after_second >= count_after_first + 2
