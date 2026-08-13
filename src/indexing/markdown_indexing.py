from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from typing import List


class MarkdwonIndexing:
    def __init__(self, max_chunk_size: int = 1000) -> None:
        self.max_chunk_size = max_chunk_size

    def index_file(self, content: str) -> List[str]:
        # 2. Setup structural splitting targets
        headers_to_split_on = [
            ("#", "Header_1"),
            ("##", "Header_2"),
            ("###", "Header_3"),
        ]

        # Instantiate the structural splitter
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            # Keep headings inside the text content.
            strip_headers=False,
        )

        # Perform Stage 1: Split by document layout and headers
        md_header_splits = markdown_splitter.split_text(content)

        # Sub-split large structural chunks for RAG limits.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=20,
            separators=["\n\n", "\n", " ", ""]
        )

        # Perform Stage 2: Combine structural splits with size limits
        final_chunks = text_splitter.split_documents(md_header_splits)

        return [chunk.page_content.strip() for chunk in final_chunks]
