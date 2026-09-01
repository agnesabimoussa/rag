from src.indexing.indexing import Indexing
from src.data_models.chunk import Chunk
from src.utils.text_processing import tokenize_text
from src.utils.file_operations import FileOperations
from typing import List
from rank_bm25 import BM25Okapi
from tqdm import tqdm


class LexicalIndexing(Indexing):
    def __init__(self,
                 chunks: List[Chunk],
                 output_file: str = "data/processed") -> None:
        super().__init__(chunks, output_file)
        self.bm25_path = self.output_file / "bm25_index.pkl"

    def create_index(self) -> BM25Okapi:
        if self.bm25_path.exists():
            bm25 = FileOperations.load_bm25(self.bm25_path)
            return bm25
        tokenized_chunks = [
            tokenize_text(self._chunk_index_text(chunk))
            for chunk in tqdm(self.chunks, desc="Tokenizing")
        ]
        bm25 = BM25Okapi(tokenized_chunks)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        FileOperations.write_bm25(self.bm25_path, bm25)
        return bm25
