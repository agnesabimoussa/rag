# subtype of Indexing
from src.indexing_module.indexing import Indexing
from src.chunking_modules.chunk import Chunk
from src.text_processing import tokenize_text
from typing import List
import pickle
from rank_bm25 import BM25Okapi
import tqdm


class LexicalIndexing(Indexing):
    def __init__(self,
                 chunks: List[Chunk],
                 output_file: str = "data/processed") -> None:
        super().__init__(chunks, output_file)

    def create_index(self) -> BM25Okapi:
        """Tokenize `self.chunks` and build/persist a BM25 index.
        Returns:
        The built `BM25Okapi` index.
        """
        tokenized_chunks = [
            tokenize_text(self._chunk_index_text(chunk))
            for chunk in tqdm(self.chunks, desc="Tokenizing")
        ]
        bm25 = BM25Okapi(tokenized_chunks)
        output_path = self.output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as file:
            pickle.dump(bm25, file)
        return bm25
