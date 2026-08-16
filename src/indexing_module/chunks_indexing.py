from typing import List
from pathlib import Path
import pickle
from rank_bm25 import BM25Okapi
from src.chunking_modules.chunk import Chunk
from tqdm import tqdm


class Indexing:
    """Builds and persists a BM25 lexical index over a list of chunks."""

    def __init__(self, chunks: List[Chunk], output_file: str) -> None:
        """Initialize the indexer.

        Args:
            chunks: Chunks to index, in the order retrieval must preserve.
            output_file: Path `create_index` pickles the BM25 index to.
        """
        self.chunks = chunks
        self.output_file = Path(output_file)

    def create_index(self) -> BM25Okapi:
        """Tokenize `self.chunks` and build/persist a BM25 index.

        Returns:
            The built `BM25Okapi` index.
        """
        tokenized_chunks = [
            chunk.text.lower().split()
            for chunk in tqdm(self.chunks, desc="Tokenizing")
        ]
        bm25 = BM25Okapi(tokenized_chunks)
        output_path = self.output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as file:
            pickle.dump(bm25, file)
        return bm25
