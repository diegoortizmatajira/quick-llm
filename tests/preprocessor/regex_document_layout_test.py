"""Tests for the RegexDocumentLayout class"""

from quick_llm.preprocessor import RegexDocumentLayout


class TestRegexDocumentLayoutInitialization:
    """Test RegexDocumentLayout initialization"""

    def test_init_with_defaults(self):
        """Test initialization with default parameters"""
        layout = RegexDocumentLayout()

        assert layout.selector_pattern == r".*"
        assert layout.line_cleaning_regex is None
        assert layout.global_cleaning_regex is None
        assert layout.skip_doc_regex is None
        assert layout.remove_empty_lines is True
        assert layout.remove_empty_docs is True
        assert layout.remove_leading_spaces is True
        assert layout.remove_trailing_spaces is True

    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters"""
        layout = RegexDocumentLayout(
            selector_pattern=r"^Document v\d+",
            line_cleaning_regex=[r"^Header", r"^Footer"],
            global_cleaning_regex=[r"<!--.*?-->"],
            skip_doc_regex=r"^SKIP THIS",
            remove_empty_lines=False,
            remove_empty_docs=False,
            remove_leading_spaces_in_lines=False,
            remove_trailing_spaces_in_lines=False,
        )

        assert layout.selector_pattern == r"^Document v\d+"
        assert layout.line_cleaning_regex == [r"^Header", r"^Footer"]
        assert layout.global_cleaning_regex == [r"<!--.*?-->"]
        assert layout.skip_doc_regex == r"^SKIP THIS"
        assert layout.remove_empty_lines is False
        assert layout.remove_empty_docs is False
        assert layout.remove_leading_spaces is False
        assert layout.remove_trailing_spaces is False

    def test_init_with_partial_parameters(self):
        """Test initialization with only some parameters specified"""
        layout = RegexDocumentLayout(
            selector_pattern=r"Manual.*",
            line_cleaning_regex=[r"Page \d+"],
        )

        assert layout.selector_pattern == r"Manual.*"
        assert layout.line_cleaning_regex == [r"Page \d+"]
        assert layout.global_cleaning_regex is None
        assert layout.skip_doc_regex is None
        # Defaults should still apply
        assert layout.remove_empty_lines is True
        assert layout.remove_empty_docs is True


class TestMatchesSelector:
    """Test matches_selector method"""

    def test_matches_selector_with_exact_match(self):
        """Test selector matching with exact pattern match"""
        layout = RegexDocumentLayout(selector_pattern=r"^Manual v\d+\.\d+$")

        assert layout.matches_selector("Manual v1.0") is True
        assert layout.matches_selector("Manual v2.5") is True

    def test_matches_selector_with_partial_match(self):
        """Test selector matching with partial pattern"""
        layout = RegexDocumentLayout(selector_pattern=r"Manual")

        assert layout.matches_selector("User Manual v1.0") is True
        assert layout.matches_selector("Manual") is True
        assert layout.matches_selector("Training Manual.pdf") is True

    def test_matches_selector_no_match(self):
        """Test selector that doesn't match the pattern"""
        layout = RegexDocumentLayout(selector_pattern=r"^Manual")

        assert layout.matches_selector("User Manual") is False
        assert layout.matches_selector("Workbook v1.0") is False

    def test_matches_selector_case_sensitive(self):
        """Test that selector matching is case-sensitive by default"""
        layout = RegexDocumentLayout(selector_pattern=r"Manual")

        assert layout.matches_selector("Manual") is True
        assert layout.matches_selector("MANUAL") is False
        assert layout.matches_selector("manual") is False

    def test_matches_selector_case_insensitive(self):
        """Test selector matching with case-insensitive pattern"""
        layout = RegexDocumentLayout(selector_pattern=r"(?i)manual")

        assert layout.matches_selector("Manual") is True
        assert layout.matches_selector("MANUAL") is True
        assert layout.matches_selector("manual") is True

    def test_matches_selector_with_default_pattern(self):
        """Test that default pattern matches everything"""
        layout = RegexDocumentLayout()

        assert layout.matches_selector("anything") is True
        assert layout.matches_selector("") is True
        assert layout.matches_selector("123 test") is True


class TestShouldSkipDoc:
    """Test should_skip_doc method"""

    def test_should_skip_empty_doc_when_enabled(self):
        """Test skipping empty documents when remove_empty_docs is True"""
        layout = RegexDocumentLayout(remove_empty_docs=True)

        assert layout.should_skip_doc("") is True

    def test_should_not_skip_empty_doc_when_disabled(self):
        """Test not skipping empty documents when remove_empty_docs is False"""
        layout = RegexDocumentLayout(remove_empty_docs=False)

        assert layout.should_skip_doc("") is False

    def test_should_skip_doc_matching_skip_regex(self):
        """Test skipping documents that match the skip_doc_regex"""
        layout = RegexDocumentLayout(skip_doc_regex=r"^Table of Contents")

        assert layout.should_skip_doc("Table of Contents\n1. Intro\n2. Chapter") is True

    def test_should_not_skip_doc_not_matching_skip_regex(self):
        """Test not skipping documents that don't match skip_doc_regex"""
        layout = RegexDocumentLayout(skip_doc_regex=r"^Table of Contents")

        assert layout.should_skip_doc("Chapter 1\nContent here") is False

    def test_should_skip_with_multiple_conditions(self):
        """Test skip logic with both empty check and regex"""
        layout = RegexDocumentLayout(
            skip_doc_regex=r"SKIP_THIS",
            remove_empty_docs=True
        )

        assert layout.should_skip_doc("") is True
        assert layout.should_skip_doc("SKIP_THIS document") is True
        assert layout.should_skip_doc("Regular content") is False

    def test_should_skip_no_skip_regex_set(self):
        """Test that documents aren't skipped when no skip_doc_regex is set"""
        layout = RegexDocumentLayout(
            skip_doc_regex=None,
            remove_empty_docs=False
        )

        assert layout.should_skip_doc("Any content") is False
        assert layout.should_skip_doc("") is False


class TestCleanContent:
    """Test clean_content method"""

    def test_clean_content_with_global_regex(self):
        """Test cleaning content with global regex patterns"""
        layout = RegexDocumentLayout(
            global_cleaning_regex=[r"<!--.*?-->", r"#\w+"]
        )

        content = "Hello <!-- comment --> World #hashtag"
        result = layout.clean_content(content)

        assert result == "Hello  World "

    def test_clean_content_with_line_regex(self):
        """Test cleaning content with line-specific regex patterns"""
        layout = RegexDocumentLayout(
            line_cleaning_regex=[r"^Header:", r"Footer$"]
        )

        content = "Header: Page 1\nContent here\nFooter"
        result = layout.clean_content(content)

        assert result == " Page 1\nContent here\n\n"

    def test_clean_content_with_both_regex_types(self):
        """Test cleaning with both global and line regex patterns"""
        layout = RegexDocumentLayout(
            global_cleaning_regex=[r"\[REDACTED\]"],
            line_cleaning_regex=[r"^Page \d+$"]
        )

        content = "Page 1\nContent [REDACTED] here\nMore content"
        result = layout.clean_content(content)

        assert result == "\nContent  here\nMore content\n"

    def test_clean_content_no_cleaning_regex(self):
        """Test that content is unchanged when no cleaning regex is set"""
        layout = RegexDocumentLayout()

        content = "Original content\nUnchanged"
        result = layout.clean_content(content)

        assert result == content

    def test_clean_content_multiple_global_patterns(self):
        """Test cleaning with multiple global regex patterns applied sequentially"""
        layout = RegexDocumentLayout(
            global_cleaning_regex=[
                r"(?i)chapter start\n-+\n",  # Remove chapter headers
                r"\d{4}-\d{2}-\d{2}",  # Remove dates
            ]
        )

        content = "Chapter start\n----------\nTitle\nDate: 2024-01-15"
        result = layout.clean_content(content)

        assert result == "Title\nDate: "

    def test_clean_content_multiple_line_patterns(self):
        """Test cleaning with multiple line regex patterns"""
        layout = RegexDocumentLayout(
            line_cleaning_regex=[
                r"^[\w\s]*Header[\w\s]*$",
                r"^[\w\s]*Footer[\w\s]*$",
            ]
        )

        content = "Main Header\nContent line\nPage Footer"
        result = layout.clean_content(content)

        assert result == "\nContent line\n\n"


class TestSkipOrProcessDocument:
    """Test skip_or_process_document method"""

    def test_skip_empty_document(self):
        """Test that empty documents are skipped when configured"""
        layout = RegexDocumentLayout(remove_empty_docs=True)

        should_skip, content = layout.skip_or_process_document("")

        assert should_skip is True
        assert content == ""

    def test_skip_document_matching_regex(self):
        """Test skipping documents that match skip_doc_regex"""
        layout = RegexDocumentLayout(skip_doc_regex=r"^CONFIDENTIAL")

        should_skip, content = layout.skip_or_process_document("CONFIDENTIAL: Internal use only")

        assert should_skip is True
        assert content == ""

    def test_process_simple_document(self):
        """Test processing a simple document without cleaning"""
        layout = RegexDocumentLayout()

        should_skip, content = layout.skip_or_process_document("Hello World")

        assert should_skip is False
        assert content == "Hello World"

    def test_process_document_with_empty_lines_removal(self):
        """Test processing with empty lines removal"""
        layout = RegexDocumentLayout(remove_empty_lines=True)

        input_content = "Line 1\n\n\nLine 2\n\n\nLine 3"
        should_skip, content = layout.skip_or_process_document(input_content)

        assert should_skip is False
        assert content == "Line 1\nLine 2\nLine 3"

    def test_process_document_with_leading_spaces_removal(self):
        """Test processing with leading spaces removal"""
        layout = RegexDocumentLayout(remove_leading_spaces_in_lines=True)

        input_content = "   Line 1\n  Line 2\nLine 3"
        should_skip, content = layout.skip_or_process_document(input_content)

        assert should_skip is False
        assert content == "Line 1\nLine 2\nLine 3"

    def test_process_document_with_trailing_spaces_removal(self):
        """Test processing with trailing spaces removal"""
        layout = RegexDocumentLayout(remove_trailing_spaces_in_lines=True)

        input_content = "Line 1   \nLine 2  \nLine 3"
        should_skip, content = layout.skip_or_process_document(input_content)

        assert should_skip is False
        assert content == "Line 1\nLine 2\nLine 3"

    def test_process_document_with_all_cleaning_options(self):
        """Test processing with all cleaning options enabled"""
        layout = RegexDocumentLayout(
            line_cleaning_regex=[r"^Header.*"],  # Remove entire line starting with Header
            global_cleaning_regex=[r"#\w+"],
            remove_empty_lines=True,
            remove_leading_spaces_in_lines=True,
            remove_trailing_spaces_in_lines=True,
        )

        input_content = "Header Line\n  Content #hashtag  \n\n\n  More content  "
        should_skip, content = layout.skip_or_process_document(input_content)

        assert should_skip is False
        assert content == "Content\nMore content"

    def test_process_document_with_cleaning_disabled(self):
        """Test processing with all cleaning options disabled"""
        layout = RegexDocumentLayout(
            remove_empty_lines=False,
            remove_leading_spaces_in_lines=False,
            remove_trailing_spaces_in_lines=False,
        )

        input_content = "  Line 1  \n\n\n  Line 2  "
        should_skip, content = layout.skip_or_process_document(input_content)

        assert should_skip is False
        # Content should be mostly preserved (only stripped at edges)
        assert "Line 1" in content
        assert "Line 2" in content

    def test_skip_after_cleaning_makes_empty(self):
        """Test that document is skipped if it becomes empty after cleaning"""
        layout = RegexDocumentLayout(
            line_cleaning_regex=[r".*"],  # Remove all content from each line
            remove_empty_docs=True,
        )

        should_skip, content = layout.skip_or_process_document("Some content")

        assert should_skip is True
        assert content == ""

    def test_process_document_with_complex_content(self):
        """Test processing a complex multi-line document with various cleaning rules"""
        layout = RegexDocumentLayout(
            line_cleaning_regex=[
                r"^[\w\s]*Header[\w\s]*$",
                r"^[\w\s]*Footer[\w\s]*$",
            ],
            global_cleaning_regex=[r"<!--.*?-->"],
            skip_doc_regex=r"^Table of Contents",
            remove_empty_lines=True,
            remove_leading_spaces_in_lines=True,
            remove_trailing_spaces_in_lines=True,
        )

        input_content = """Document Header


Content paragraph <!-- inline comment -->

More content

    Document Footer"""

        should_skip, content = layout.skip_or_process_document(input_content)

        assert should_skip is False
        assert "Content paragraph" in content
        assert "More content" in content
        assert "Header" not in content
        assert "Footer" not in content
        assert "comment" not in content


class TestGetMatchingLayout:
    """Test get_matching_layout static method"""

    def test_get_matching_layout_first_match(self):
        """Test getting the first layout that matches the selector"""
        layouts = [
            RegexDocumentLayout(selector_pattern=r"Manual"),
            RegexDocumentLayout(selector_pattern=r"Workbook"),
            RegexDocumentLayout(selector_pattern=r".*"),  # Catch-all
        ]

        result = RegexDocumentLayout.get_matching_layout(layouts, "User Manual v1.0")

        assert result is layouts[0]

    def test_get_matching_layout_fallback_to_catchall(self):
        """Test falling back to catch-all layout when specific ones don't match"""
        layouts = [
            RegexDocumentLayout(selector_pattern=r"^Manual"),
            RegexDocumentLayout(selector_pattern=r"^Workbook"),
            RegexDocumentLayout(selector_pattern=r".*"),  # Catch-all
        ]

        result = RegexDocumentLayout.get_matching_layout(layouts, "Guide v1.0")

        assert result is layouts[2]

    def test_get_matching_layout_no_match(self):
        """Test getting None when no layout matches"""
        layouts = [
            RegexDocumentLayout(selector_pattern=r"^Manual"),
            RegexDocumentLayout(selector_pattern=r"^Workbook"),
        ]

        result = RegexDocumentLayout.get_matching_layout(layouts, "Guide v1.0")

        assert result is None

    def test_get_matching_layout_empty_collection(self):
        """Test getting None from empty layout collection"""
        result = RegexDocumentLayout.get_matching_layout([], "Any selector")

        assert result is None

    def test_get_matching_layout_multiple_matches_returns_first(self):
        """Test that when multiple layouts match, the first one is returned"""
        layout1 = RegexDocumentLayout(selector_pattern=r"Manual")
        layout2 = RegexDocumentLayout(selector_pattern=r"User Manual")
        layout3 = RegexDocumentLayout(selector_pattern=r".*")

        layouts = [layout1, layout2, layout3]

        result = RegexDocumentLayout.get_matching_layout(layouts, "User Manual v1.0")

        # Should return the first matching layout (layout1)
        assert result is layout1

    def test_get_matching_layout_with_complex_patterns(self):
        """Test matching with complex regex patterns"""
        layouts = [
            RegexDocumentLayout(selector_pattern=r"Manual v(\d+\.)+\d+"),
            RegexDocumentLayout(selector_pattern=r"Workbook v(\d+\.)+\d+"),
            RegexDocumentLayout(selector_pattern=r".*\.pdf$"),
        ]

        result = RegexDocumentLayout.get_matching_layout(layouts, "Manual v1.2.3")
        assert result is layouts[0]

        result = RegexDocumentLayout.get_matching_layout(layouts, "Workbook v2.0.1")
        assert result is layouts[1]

        result = RegexDocumentLayout.get_matching_layout(layouts, "Unknown.pdf")
        assert result is layouts[2]


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple features"""

    def test_pdf_manual_processing_scenario(self):
        """Test a realistic PDF manual processing scenario"""
        layout = RegexDocumentLayout(
            selector_pattern=r"Manual v\d+",
            line_cleaning_regex=[
                r"^\s*Page \d+\s*$",  # Remove page numbers (with optional whitespace)
                r"^\s*[-]+\s*$",  # Remove separator lines
            ],
            global_cleaning_regex=[
                r"Copyright.*?All rights reserved\.",  # Remove copyright notices
            ],
            skip_doc_regex=r"^(Table of Contents|Index)",
            remove_empty_lines=True,
            remove_leading_spaces_in_lines=True,
            remove_trailing_spaces_in_lines=True,
        )

        # Test selector matching
        assert layout.matches_selector("Manual v1.0") is True

        # Test skipping table of contents
        toc_content = "Table of Contents\n1. Introduction\n2. Getting Started"
        should_skip, _ = layout.skip_or_process_document(toc_content)
        assert should_skip is True

        # Test processing normal content
        content = """Page 1
-------------

Chapter Title

This is the content. Copyright 2024. All rights reserved.

More content here.

Page 2"""

        should_skip, result = layout.skip_or_process_document(content)
        assert should_skip is False
        assert "Chapter Title" in result
        assert "This is the content." in result
        assert "More content here." in result
        assert "Page 1" not in result
        assert "Page 2" not in result
        assert "Copyright" not in result
        assert "-----" not in result

    def test_workbook_processing_scenario(self):
        """Test a workbook document processing scenario"""
        layout = RegexDocumentLayout(
            selector_pattern=r"Workbook",
            line_cleaning_regex=[r"Page\s+\d+$"],
            global_cleaning_regex=[r"(?i)chapter start\n-+\n"],
            remove_empty_lines=True,
        )

        content = """Chapter start
-------------------
Chapter 1: Introduction

This is the chapter content.

Page 1"""

        should_skip, result = layout.skip_or_process_document(content)

        assert should_skip is False
        assert "Chapter 1: Introduction" in result
        assert "This is the chapter content." in result
        assert "Chapter start" not in result
        assert "Page 1" not in result
        assert "---" not in result

    def test_generic_document_processing_scenario(self):
        """Test generic document processing with default catch-all layout"""
        layout = RegexDocumentLayout(
            line_cleaning_regex=[
                r"^[\w\s]*\s-\s[\w\s]*$",  # Headers with dash separator
                r"(?i)\s*pa?ge?\.?\s+\d+\s*$",  # Various page number formats
            ],
            global_cleaning_regex=[r"#\w+"],  # Remove hashtags
        )

        content = """Document Title - Section Header

Content with #hashtag and more text.

Additional content.

Pg. 5"""

        should_skip, result = layout.skip_or_process_document(content)

        assert should_skip is False
        assert "Content with  and more text." in result
        assert "Additional content." in result
        assert "Section Header" not in result
        assert "#hashtag" not in result
        assert "Pg. 5" not in result
