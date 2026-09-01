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

    def create_index(self, force: bool = True) -> BM25Okapi:
        """Rebuild the BM25 index, or reuse the cached one on disk.

        BM25's IDF statistics are corpus-global, so unlike the vector index
        there's no way to patch just the changed chunks in place: any change
        to the corpus requires a full rebuild from `self.chunks` (the caller
        is expected to pass the already-updated full chunk list). Set
        `force=False` when the caller has determined nothing changed, to
        skip the rebuild entirely and load the cached pickle.
        """
        if not force and self.bm25_path.exists():
            return FileOperations.load_bm25(self.bm25_path)
        tokenized_chunks = [
            tokenize_text(self._chunk_index_text(chunk))
            for chunk in tqdm(self.chunks, desc="Tokenizing")
        ]
        bm25 = BM25Okapi(tokenized_chunks)
        self.output_file.mkdir(parents=True, exist_ok=True)
        FileOperations.write_bm25(self.bm25_path, bm25)
        return bm25
