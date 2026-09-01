from typing import List, Any
from pathlib import Path
from src.data_models.chunk import (Chunk, MarkdownChunk, CodeChunk)
from abc import ABC, abstractmethod


class Indexing(ABC):
    """Builds and persists a BM25 lexical index over a list of chunks."""

    def __init__(self,
                 chunks: List[Chunk],
                 output_file: str) -> None:
        """Initialize the indexer.

        Args:
            chunks: Chunks to index, in the order retrieval must preserve.
            output_file: Path `create_index` pickles the BM25 index to.
        """
        self.chunks = chunks
        self.output_file = Path(output_file)

    @staticmethod
    def _chunk_index_text(chunk: Chunk) -> str:
        index_text = chunk.text
        if isinstance(chunk, MarkdownChunk) and chunk.section:
            index_text = f"{chunk.section}\n{index_text}"
        if isinstance(chunk, CodeChunk) and chunk.type:
            index_text = f"{chunk.type}\n{index_text}"
        return index_text

    @abstractmethod
    def create_index(self) -> Any:
        pass
