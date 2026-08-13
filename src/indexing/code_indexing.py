# block chunking (function / class)
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from typing import List


class CodeIndexing:
    def __init__(self,
                 max_chunk_size: int | None = None) -> None:
        self.max_chunk_size = max_chunk_size

    def index_file(self, content: str) -> List[str]:
        chunks = []
        # 2. Initialize the splitter optimized for Python syntax
        # Chunk size is small here to force clean splits along boundaries
        if self.max_chunk_size:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                chunk_size=self.max_chunk_size,
                chunk_overlap=20
            )
        else:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                chunk_overlap=20
            )
        # 3. Create the document chunks
        chunks = splitter.create_documents([content])
        # 4. Display the structural chunks
        for i, chunk in enumerate(chunks):
            chunks.append(chunk)
        return chunks
