# block chunking (function / class)
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from typing import List


class CodeChunking:
    def __init__(self,
                 max_chunk_size: int = 2000) -> None:
        self.max_chunk_size = max_chunk_size

    def chunk_file(self, content: str) -> List[str]:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=self.max_chunk_size,
            chunk_overlap=20
        )
        documents = splitter.create_documents([content])
        return [document.page_content.strip() for document in documents]
