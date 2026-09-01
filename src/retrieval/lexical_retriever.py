from rank_bm25 import BM25Okapi
import heapq
from src.data_models.minimal_source import MinimalSource
from src.utils.text_processing import tokenize_text
from typing import List
from src.data_models.chunk import Chunk


class LexicalRetriever:
    def __init__(self,
                 bm25: BM25Okapi,
                 chunks: List[Chunk],
                 k: int) -> None:
        self.bm25 = bm25
        self.chunks = chunks
        self.k = k

    def retrieve_context(self, prompt: str) -> List[MinimalSource]:
        tokenized_query = tokenize_text(prompt)
        if not tokenized_query:
            tokenized_query = prompt.lower().split()

        scores = self.bm25.get_scores(tokenized_query)
        candidate_pool = min(len(self.chunks), max(self.k * 8, 40))
        ranked_indices = [
            index
            for index, _ in heapq.nlargest(
                candidate_pool,
                enumerate(scores),
                key=lambda item: item[1],
            )
        ]
        ranked_chunks = [self.chunks[index] for index in ranked_indices]

        unique_chunks = []
        seen_groups = set()
        seen_chunk_ids = set()
        seen_source_spans = set()

        for chunk in ranked_chunks:
            group_id = chunk.original_chunk_id or chunk.id
            if group_id in seen_groups:
                continue
            source_span = (
                chunk.source,
                chunk.first_character_index,
                chunk.last_character_index,
            )
            if chunk.id not in seen_chunk_ids and source_span not in seen_source_spans:
                unique_chunks.append(chunk)
                seen_groups.add(group_id)
                seen_chunk_ids.add(chunk.id)
                seen_source_spans.add(source_span)
            if len(unique_chunks) == self.k:
                break

        if len(unique_chunks) < self.k:
            for chunk in ranked_chunks:
                source_span = (
                    chunk.source,
                    chunk.first_character_index,
                    chunk.last_character_index,
                )
                if chunk.id in seen_chunk_ids or source_span in seen_source_spans:
                    continue
                unique_chunks.append(chunk)
                seen_chunk_ids.add(chunk.id)
                seen_source_spans.add(source_span)
                if len(unique_chunks) == self.k:
                    break

        return [
            MinimalSource(
                file_path=chunk.source,
                first_character_index=chunk.first_character_index,
                last_character_index=chunk.last_character_index,
                scope=getattr(chunk, "type", None)
            )
            for chunk in unique_chunks
        ]
