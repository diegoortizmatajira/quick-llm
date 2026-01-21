"""
This module defines a Preprocessor middleware that can be used along with any
other loader to executing a preprocessing cleanup based on regular expressions
"""

from typing import Callable, Iterator

from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document

from .regex_document_layout import RegexDocumentLayout


class RegexPreprocessorLoader(BaseLoader):
    """
    A document loader middleware that preprocesses documents using customizable
    regular expression-based layouts. This extends the functionality of a source
    document loader by applying preprocessing rules defined in the layouts.

    The preprocessing involves transforming the content of the documents or skipping
    some documents entirely based on the rules defined in the provided layouts.

    Attributes:
        source_loader (BaseLoader): The original document loader whose output will
            be preprocessed.
        layout_list (list[RegexDocumentLayout]): A collection of layouts specifying
            regex patterns and preprocessing rules.
        layout_selector (Callable[[Document], str]): A function that determines which
            layout from the layout_list should be used for a given document.
    """

    def __init__(
        self,
        source_loader: BaseLoader,
        layout_list: list[RegexDocumentLayout],
        layout_selector: Callable[[Document], str],
    ):
        """
        Initializes the RegexPreprocessorLoader.

        Args:
            source_loader (BaseLoader): The original document loader whose output will be processed.
            layout_list (list[RegexDocumentLayout]): A list of RegexDocumentLayout objects used to
                define the preprocessing patterns and rules.
            layout_selector (Callable[[Document], str]): A callable that takes a Document and returns
                a selector value used to match the appropriate layout from the layout_list.
        """
        self.source_loader = source_loader
        self.layout_list = layout_list
        self.layout_selector = layout_selector

    def lazy_load(self) -> Iterator[Document]:
        """
        Processes documents from the source loader, applies the regex preprocessing
        based on matching layouts, and yields the processed documents.

        - For each document obtained from the source loader, the `layout_selector`
          is used to determine the appropriate layout from the `layout_list`.
        - If a matching layout is found, its `skip_or_process_document` method
          is invoked to preprocess the document content and decide whether the
          document should be skipped.
        - Only non-skipped documents are yielded.

        Returns:
            Iterator[Document]: An iterator over the processed documents.
        """
        for doc in self.source_loader.load():
            # gets the first layout that matches the file pattern
            selector_value = self.layout_selector(doc)
            layout = RegexDocumentLayout.get_matching_layout(
                self.layout_list, selector_value
            )
            if layout:
                # If there is a layout, then process the doc and check if it should be skipped
                should_skip, doc.page_content = layout.skip_or_process_document(
                    doc.page_content
                )
                if should_skip:
                    continue
            yield doc
