"""Tests for the RegexPreprocessorLoader with various document layouts and cleaning rules."""

from pytest_mock import MockFixture
from langchain_core.documents import Document

from quick_llm.preprocessor import RegexDocumentLayout, RegexPreprocessorLoader


LAYOUTS = [
    RegexDocumentLayout(
        selector_pattern=r"Manual v(\.\d+)+",
        line_cleaning_regex=[
            r"^[\w\s]*Header[\w\s]*$",  # Headers
            r"^[\w\s]*Footer[\w\s]*$",  # Footer
        ],
        skip_doc_regex=r"^Table of Contents",
    ),
    RegexDocumentLayout(
        selector_pattern=r"Workbook v(\.\d+)+",
        line_cleaning_regex=[
            r"Page\s\d+$"  # Footers with page number only
        ],
        global_cleaning_regex=[
            r"(?i)chapter start\n-+\n"  # Case-insensitive Multi-line header with multiple dashes in the second line
        ],
    ),
    # Default for unclassified layouts
    RegexDocumentLayout(
        line_cleaning_regex=[
            r"^[\w\s]*\s-\s[\w\s]*$",  # Headers with words spaces and a single dash as separator
            r"^[-\s]+$",  # Separator dashed line
            r"(?i)\s*pa?ge?\.?\s+\d+\s*$",  # Footers with "page", "pag.", "pg." and a number.
        ],
        global_cleaning_regex=[
            r"#\w+"  # Removes hashtags
        ],
    ),
]


def test_direct_chat_provider(mocker: MockFixture):
    """
    Tests the direct chat provider functionality with a mocked data loader and regex layouts.
    Ensures that the preprocessing logic applies the appropriate cleaning and skipping rules
    based on the document layout configurations.

    Args:
        mocker (MockFixture): The pytest-mock fixture used to create mock objects.
    """
    test_documents = [
        Document(
            """
Table of Contents 
1.Intro
2. Content
This is a Footer",
""",
            metadata={"filename": "User Manual v.1.0.pdf", "page": 1},
        ),
        Document(
            """
This is a Header Line
Here is the content
This is a Footer
""",
            metadata={"filename": "User Manual v.1.0.pdf", "page": 2},
        ),
        Document(
            """
This is a Header Line
Here is the content
Page 1
""",
            metadata={"filename": "Student Workbook v.2.pdf", "page": 1},
        ),
        Document(
            """
Chapter start
-------------------
Chapter Title
Here is the content
Page 2
""",
            metadata={"filename": "Student Workbook v.2.pdf", "page": 2},
        ),
        Document(
            """
Student Guide - For students only
- - - - - -
Here is the content #hashtag
- - - - - -
Here is more content
Pg. 2
""",
            metadata={"filename": "Student Guide.pdf", "page": 1},
        ),
    ]
    # Create a mock loader that returns the test documents
    mock_loader = mocker.MagicMock()
    mock_loader.load.return_value = test_documents
    # Create the RegexPreprocessorLoader with the mock loader and layouts
    prepocessor = RegexPreprocessorLoader(
        mock_loader, LAYOUTS, lambda d: d.metadata["filename"]
    )
    docs = prepocessor.load()
    assert docs is not None
    # This doc is the User Manual
    assert (
        docs[0].metadata["page"] == 2
    )  # First page is skipped as it starts with "Table of content"
    assert (
        docs[0].page_content == "Here is the content"
    )  # header and footer are removed
    # This doc is the Student Workbook
    assert (
        docs[1].page_content == "This is a Header Line\nHere is the content"
    )  # only footer is removed
    assert docs[2].page_content == "Chapter Title\nHere is the content"
    # This doc is the Student Guide that matches the generic Layout
    assert docs[3].page_content == "Here is the content\nHere is more content"
