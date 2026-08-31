# subtype of Indexing
from src.models.embedding_model import EmbeddingModel
from src.indexing_module.chunks_indexing import Indexing
from src.chunking_modules.chunk import Chunk
import chromadb
from typing import List


class VectorIndexing(Indexing):
    def __init__(self,
                 chunks: List[Chunk],
                 output_file: str = "data/processed") -> None:
        super.__init__(chunks, output_file)
        self.embedding_model = EmbeddingModel()
        self.client = chromadb.PersistentClient(path=f"{output_file}/chroma_db")
        self.collection = self.client.get_or_create_collection(name="vllm documents")

    def create_index(self) -> None:
        chunks = [self._chunk_index_text(chunk) for chunk in self.chunks]
        ids = [chunk.id for chunk in chunks]
        embeddings = self.embedding_model.embed_chunks(chunks)
        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist(),
        )
