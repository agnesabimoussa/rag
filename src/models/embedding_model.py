from src.models.model_download import ModelDownload
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List


class EmbeddingModel:
    def __init__(self, model_path: str = "sentence-transformers/all-MiniLM-L6-v2"):
        local_weights_dir = ModelDownload._ensure_local_weights(model_path)
        self.embedding_model = SentenceTransformer(local_weights_dir)

    def embed_text(self, text: str) -> np.ndarray:
        embedding = self.embedding_model.encode(text)
        return embedding

    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        embeddings = self.embedding_model.encode([chunk for chunk in chunks])
        return embeddings
