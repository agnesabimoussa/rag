"""Shared post-ranking selection used by every retriever.

Turns an already-ranked list of chunks into the final top-k `MinimalSource`
list: chunks from the same split group (`original_chunk_id`) or covering the
exact same source span are collapsed to one result, so a question doesn't get
answered with three overlapping slices of the same paragraph/function.
"""
from typing import List, Set, Tuple

from src.data_models.chunk import Chunk
from src.data_models.minimal_source import MinimalSource


def select_diverse_sources(ranked_chunks: List[Chunk], k: int) -> List[MinimalSource]:
    if k <= 0:
        return []

    unique_chunks: List[Chunk] = []
    seen_groups: Set[str] = set()
    seen_chunk_ids: Set[str] = set()
    seen_source_spans: Set[Tuple[str, int, int]] = set()

    for chunk in ranked_chunks:
        group_id = chunk.original_chunk_id or chunk.id
        if group_id in seen_groups:
            continue
        source_span = (chunk.source, chunk.first_character_index, chunk.last_character_index)
        if chunk.id not in seen_chunk_ids and source_span not in seen_source_spans:
            unique_chunks.append(chunk)
            seen_groups.add(group_id)
            seen_chunk_ids.add(chunk.id)
            seen_source_spans.add(source_span)
        if len(unique_chunks) == k:
            break

    if len(unique_chunks) < k:
        for chunk in ranked_chunks:
            source_span = (chunk.source, chunk.first_character_index, chunk.last_character_index)
            if chunk.id in seen_chunk_ids or source_span in seen_source_spans:
                continue
            unique_chunks.append(chunk)
            seen_chunk_ids.add(chunk.id)
            seen_source_spans.add(source_span)
            if len(unique_chunks) == k:
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
