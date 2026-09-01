from src.data_models.chunk import Chunk
from src.data_models.minimal_source import MinimalSource
from src.models.embeddings import EmbeddingModel
from src.retrieval.ranking import select_diverse_sources
from typing import Dict, List, Optional
from chromadb import Collection


class SemanticRetriever:
    def __init__(self,
                 collection: Collection,
                 k: int) -> None:
        self.embedding_model = EmbeddingModel()
        self.collection = collection
        self.k = k

    def rank_chunk_ids(self, prompt: str, pool_size: Optional[int] = None) -> List[str]:
        """Return chunk IDs ranked by embedding similarity, best first."""
        pool_size = pool_size if pool_size is not None else max(self.k * 8, 40)
        embedding = self.embedding_model.embed_text(prompt)
        results = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=pool_size,
        )
        ids = results.get("ids") or [[]]
        return list(ids[0])

    def retrieve_context(self, prompt: str, chunks_by_id: Dict[str, Chunk]) -> List[MinimalSource]:
        ranked_chunks = [chunks_by_id[chunk_id] for chunk_id in self.rank_chunk_ids(prompt)
                         if chunk_id in chunks_by_id]
        return select_diverse_sources(ranked_chunks, self.k)
