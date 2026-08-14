from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from typing import List


class MarkdwonChunking:
    def __init__(self, max_chunk_size: int = 2000) -> None:
        self.max_chunk_size = max_chunk_size

    def chunk_file(self, content: str) -> List[str]:
        headers_to_split_on = [
            ("#", "Header_1"),
            ("##", "Header_2"),
            ("###", "Header_3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )
        md_header_splits = markdown_splitter.split_text(content)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=20,
            separators=["\n\n", "\n", " ", ""]
        )
        final_chunks = text_splitter.split_documents(md_header_splits)
        return [chunk.page_content.strip() for chunk in final_chunks]
