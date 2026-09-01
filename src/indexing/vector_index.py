from src.models.embeddings import EmbeddingModel
from src.indexing.indexing import Indexing
from src.data_models.chunk import Chunk
import chromadb
from typing import List
from chromadb import Collection


class VectorIndexing(Indexing):
    def __init__(self,
                 chunks: List[Chunk],
                 output_file: str = "data/processed") -> None:
        super().__init__(chunks, output_file)
        self.embedding_model = EmbeddingModel()
        self.chroma_path = f"{output_file}/chroma_db"
        self.collection_name = "documents"
        self.client = chromadb.PersistentClient(path=self.chroma_path)

    def create_index(self) -> Collection:
        try:
            collection = self.client.get_collection(name=self.collection_name)
            return collection
        except Exception:
            collection = self.client.create_collection(name=self.collection_name)
            batch_size = 1000
            chunks = [self._chunk_index_text(chunk) for chunk in self.chunks]
            ids = [chunk.id for chunk in self.chunks]
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]
                embeddings = self.embedding_model.embed_chunks(batch_chunks)
                collection.add(
                    ids=batch_ids,
                    documents=batch_chunks,
                    embeddings=embeddings.tolist(),
                )
            return collection
