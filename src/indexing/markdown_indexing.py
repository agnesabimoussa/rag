from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from typing import List


class MarkdwonIndexing:
    def __init__(self, max_chunk_size):
        self.max_chunk_size = max_chunk_size

    def index_file(self, content: str) -> List[str]:
        chunks = []
        # 2. Setup structural splitting targets
        # This mapping dictates which markdown headers to split on and what metadata keys to map them to.
        headers_to_split_on = [
            ("#", "Header_1"),
            ("##", "Header_2"),
            ("###", "Header_3"),
        ]

        # Instantiate the structural splitter
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False  # Set to False to keep headings inside the text content as well
        )

        # Perform Stage 1: Split by document layout and headers
        md_header_splits = markdown_splitter.split_text(content)

        # 3. Setup Stage 2: Sub-splitting large structural chunks for RAG limits
        # This ensures sections with massive blocks of text don't overflow your embedding window.
        text_splitter = RecursiveCharacterTextSplitter(
            # Targeted length per sub-chunk (in characters)
            chunk_size=self.max_chunk_size,
            chunk_overlap=20,     # Context overlap between consecutive sub-chunks
            # Favors paragraphs and sentences over word breaks
            separators=["\n\n", "\n", " ", ""]
        )

        # Perform Stage 2: Combine structural splits with size limits
        final_chunks = text_splitter.split_documents(md_header_splits)

        # 4. View your results
        for i, chunk in enumerate(final_chunks):
            chunks.append(chunk.page_content.strip())
        return chunks
