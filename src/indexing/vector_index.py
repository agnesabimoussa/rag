from src.models.embeddings import EmbeddingModel
from src.indexing.indexing import Indexing
from src.data_models.chunk import Chunk
import chromadb
from typing import List, Optional
from chromadb import Collection


class VectorIndexing(Indexing):
    def __init__(self,
                 chunks: List[Chunk],
                 output_file: str = "data/processed",
                 removed_ids: Optional[List[str]] = None) -> None:
        """Incrementally update the Chroma collection.

        `chunks` should be only the new/changed chunks to embed (not the
        whole corpus) and `removed_ids` the IDs of chunks whose file was
        changed or deleted, so their stale vectors get dropped. Chroma
        supports per-ID upsert/delete, so unchanged chunks are never
        re-embedded or re-written.
        """
        super().__init__(chunks, output_file)
        self.chroma_path = f"{output_file}/chroma_db"
        self.collection_name = "documents"
        self.client = chromadb.PersistentClient(path=self.chroma_path)
        self.removed_ids = removed_ids or []

    def create_index(self) -> Collection:
        collection = self.client.get_or_create_collection(name=self.collection_name)
        if self.removed_ids:
            collection.delete(ids=self.removed_ids)
        if self.chunks:
            embedding_model = EmbeddingModel()
            batch_size = 1000
            texts = [self._chunk_index_text(chunk) for chunk in self.chunks]
            ids = [chunk.id for chunk in self.chunks]
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]
                embeddings = embedding_model.embed_chunks(batch_texts)
                collection.upsert(
                    ids=batch_ids,
                    documents=batch_texts,
                    embeddings=embeddings.tolist(),
                )
        return collection
