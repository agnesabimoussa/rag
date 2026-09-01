from rank_bm25 import BM25Okapi
import heapq
from src.data_models.minimal_source import MinimalSource
from src.retrieval.ranking import select_diverse_sources
from src.utils.text_processing import tokenize_text
from typing import List, Optional
from src.data_models.chunk import Chunk


class LexicalRetriever:
    def __init__(self,
                 bm25: BM25Okapi,
                 chunks: List[Chunk],
                 k: int) -> None:
        self.bm25 = bm25
        self.chunks = chunks
        self.k = k

    def _default_pool_size(self) -> int:
        return min(len(self.chunks), max(self.k * 8, 40))

    def rank_chunk_ids(self, prompt: str, pool_size: Optional[int] = None) -> List[str]:
        """Return chunk IDs ranked by BM25 score, best first."""
        tokenized_query = tokenize_text(prompt)
        if not tokenized_query:
            tokenized_query = prompt.lower().split()

        scores = self.bm25.get_scores(tokenized_query)
        candidate_pool = min(len(self.chunks), pool_size if pool_size is not None
                             else self._default_pool_size())
        ranked_indices = [
            index
            for index, _ in heapq.nlargest(
                candidate_pool,
                enumerate(scores),
                key=lambda item: item[1],
            )
        ]
        return [self.chunks[index].id for index in ranked_indices]

    def retrieve_context(self, prompt: str) -> List[MinimalSource]:
        chunks_by_id = {chunk.id: chunk for chunk in self.chunks}
        ranked_chunks = [chunks_by_id[chunk_id] for chunk_id in self.rank_chunk_ids(prompt)]
        return select_diverse_sources(ranked_chunks, self.k)
