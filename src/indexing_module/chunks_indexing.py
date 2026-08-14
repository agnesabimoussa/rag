from typing import List
from pathlib import Path
import pickle
from rank_bm25 import BM25Okapi


class Indexing:
    def __init__(self, chunks: List[str], output_file: str) -> None:
        self.chunks = chunks
        self.output_file = Path(output_file)

    def create_index(self) -> None:
        tokenized_chunks = [
            chunk.lower().split()
            for chunk in self.chunks
        ]
        bm25 = BM25Okapi(tokenized_chunks)
        output_path = self.output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as file:
            pickle.dump(bm25, file)
