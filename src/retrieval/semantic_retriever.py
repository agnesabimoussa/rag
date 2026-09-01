from src.data_models.minimal_source import MinimalSource
from src.models.embeddings import EmbeddingModel
from typing import List
from chromadb import Collection


class SemanticRetriever:
    def __init__(self,
                 collection: Collection,
                 k: int) -> None:
        self.embedding_model = EmbeddingModel()
        self.collection = collection

    def retrieve_context(self, prompt: str) -> List[MinimalSource]:
        embedding = self.embedding_model.embed_text(prompt)
        results = self.collection.query(
            query_embeddings=embedding,  # Pass the raw list of floats
            n_results=self.k,
            include=["documents", "metadatas", "distances"]
        )
        return results
