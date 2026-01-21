"""
This module defines the class that defines the rules to process a given document layout
"""

import re

from typing import Optional, Tuple


class RegexDocumentLayout:
    """
    A class that defines the rules to process a given document layout.

    This class provides functionality for selecting, processing, and cleaning documents using
    regular expressions. It also facilitates skipping documents based on defined conditions and
    retrieving document layouts that match specific selectors.

    Attributes:
        selector_pattern (str): Regular expression for checking if an instance matches a selector value.
        line_cleaning_regex (Optional[list[str]]): Regular expressions for removing lines where a match is found.
        global_cleaning_regex (Optional[list[str]]): Regular expressions for removing multi-line content.
        skip_doc_regex (Optional[str]): Regular expression for skipping certain documents.
        remove_empty_lines (bool): Whether to remove empty lines during preprocessing.
        remove_empty_docs (bool): Whether to remove empty documents during preprocessing.
        remove_leading_spaces (bool): Whether to remove leading spaces in lines during preprocessing.
        remove_trailing_spaces (bool): Whether to remove trailing spaces in lines during preprocessing.
    """

    def __init__(
        self,
        *,
        selector_pattern: str = r".*",
        line_cleaning_regex: Optional[list[str]] = None,
        global_cleaning_regex: Optional[list[str]] = None,
        skip_doc_regex: Optional[str] = None,
        remove_empty_lines: bool = True,
        remove_empty_docs: bool = True,
        remove_leading_spaces_in_lines: bool = True,
        remove_trailing_spaces_in_lines: bool = True,
    ):
        """
        Initializes a RegexDocumentLayout instance with the provided configuration.

        Args:
            selector_pattern (str): Regular expression used to check if an instance matches a selector.
            line_cleaning_regex (Optional[list[str]]): List of regex patterns for cleaning lines that match.
            global_cleaning_regex (Optional[list[str]]): List of regex patterns for cleaning multi-line content.
            skip_doc_regex (Optional[str]): Regex pattern that determines if a document should be skipped.
            remove_empty_lines (bool): If True, removes empty lines during preprocessing.
            remove_empty_docs (bool): If True, skips documents that are empty.
            remove_leading_spaces_in_lines (bool): If True, removes leading spaces from lines during preprocessing.
            remove_trailing_spaces_in_lines (bool): If True, removes trailing spaces from lines during preprocessing.
        """
        self.selector_pattern = selector_pattern
        self.line_cleaning_regex = line_cleaning_regex
        self.global_cleaning_regex = global_cleaning_regex
        self.skip_doc_regex = skip_doc_regex
        self.remove_empty_lines = remove_empty_lines
        self.remove_empty_docs = remove_empty_docs
        self.remove_leading_spaces = remove_leading_spaces_in_lines
        self.remove_trailing_spaces = remove_trailing_spaces_in_lines

    def matches_selector(self, selector: str) -> bool:
        """
        Determines if the given selector matches the layout's selector pattern.

        Args:
            selector (str): The selector to check against the layout's pattern.

        Returns:
            bool: True if the selector matches the pattern, False otherwise.
        """
        return re.search(self.selector_pattern, selector) is not None

    def should_skip_doc(self, doc_content: str) -> bool:
        """
        Determines whether a document should be skipped based on its content.

        Args:
            doc_content (str): The content of the document to evaluate.

        Returns:
            bool: True if the document should be skipped, False otherwise.
        """
        if self.remove_empty_docs and doc_content == "":
            return True
        if self.skip_doc_regex:
            return re.search(self.skip_doc_regex, doc_content) is not None
        return False

    def clean_content(self, doc_content: str) -> str:
        """
        Cleans the content of a document based on the defined cleaning rules.

        This method applies both global and line-specific cleaning regular expressions to 
        modify the content of the document. Global cleaning is applied over the entire 
        document, and line-specific cleaning processes each line separately. Processed lines 
        are then reassembled into the final cleaned document.

        Args:
            doc_content (str): The original content of the document.

        Returns:
            str: The cleaned content of the document.
        """
        result = doc_content
        if self.global_cleaning_regex:
            for remove_regex in self.global_cleaning_regex:
                result = re.sub(remove_regex, "", result)
        if self.line_cleaning_regex:
            full_doc = result
            # it will reassemble the document line by line with clean lines
            result = ""
            for line in full_doc.split("\n"):
                for remove_regex in self.line_cleaning_regex:
                    line = re.sub(remove_regex, "", line)
                result += line + "\n"
        return result

    def skip_or_process_document(self, doc_content: str) -> Tuple[bool, str]:
        """
        Determines whether to skip or process a given document and cleans its content.

        This method evaluates the document's content to decide if it should be skipped 
        based on the defined conditions. If the document is not skipped, its content is 
        cleaned according to the rules specified for the layout, such as removing empty 
        lines, leading/trailing spaces, and applying regular expressions for line/global 
        cleaning.

        Args:
            doc_content (str): The content of the document to evaluate and process.

        Returns:
            Tuple[bool, str]: A tuple where the first element is a boolean indicating if 
                              the document should be skipped, and the second element is 
                              the cleaned document content (or an empty string if skipped).
        """
        result = doc_content
        # Before any processing:
        # if the layout identifies it should skip the page, then exit
        if self.should_skip_doc(result):
            return True, ""
        result = self.clean_content(result)
        if self.remove_empty_lines:
            result = re.sub(r"\n([\s\t]*\n)+", "\n", result)
        if self.remove_leading_spaces:
            result = re.sub(r"(^|(?<=\n))\s+", "", result)
        if self.remove_trailing_spaces:
            result = re.sub(r"\s+($|(?=\n))", "", result)
        result = result.strip()

        # After cleaning process:
        # if the layout identifies it should skip the page, the jump to the next iteration
        if self.should_skip_doc(result):
            return True, ""
        return False, result

    @staticmethod
    def get_matching_layout(
        layout_collection: list["RegexDocumentLayout"], selector: str
    ) -> Optional["RegexDocumentLayout"]:
        """
        Retrieves the first layout that matches the given selector from a collection of layouts.

        This method iterates over a collection of RegexDocumentLayout instances and 
        returns the first one whose selector pattern matches the provided selector.

        Args:
            layout_collection (list["RegexDocumentLayout"]): A list of RegexDocumentLayout instances to evaluate.
            selector (str): The selector string to match against the layouts' selector patterns.

        Returns:
            Optional["RegexDocumentLayout"]: The first RegexDocumentLayout that matches the selector, 
                                             or None if no match is found.
        """
        matching_layouts = [
            layout for layout in layout_collection if layout.matches_selector(selector)
        ]
        if len(matching_layouts) > 0:
            return matching_layouts[0]
        return None
