import itertools
from typing import List
from pathlib import Path
from abc import ABC, abstractmethod
from src.chunking_modules.chunk import Chunk


class AbstractChunker(ABC):
    def __init__(self, max_chunk_size: int) -> None:
        self.id_generator = itertools.count(start=1)
        self.max_chunk_size = max_chunk_size

    def locate_span(self,
                    source_text: str,
                    target_text: str,
                    cursor: int) -> tuple[int, int, int]:
        start = source_text.find(target_text, cursor)
        if start == -1:
            start = source_text.find(target_text)
        if start == -1:
            start = cursor
        end = start + len(target_text)
        next_cursor = max(cursor, end)
        return start, end, next_cursor

    def locate_subchunk_spans(self,
                              section_text: str,
                              sub_texts: list[str],
                              section_start: int) -> list[tuple[str, int, int]]:
        spans: list[tuple[str, int, int]] = []
        section_cursor = 0
        for text in sub_texts:
            sub_start_relative, sub_end_relative, section_cursor = self.locate_span(
                section_text,
                text,
                section_cursor,
            )
            sub_start = section_start + sub_start_relative
            sub_end = section_start + sub_end_relative
            spans.append((text, sub_start, sub_end))
        return spans

    @staticmethod
    def read_file(file: Path) -> str:
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
        return content

    def chunk_files(self,
                    files: List[Path]) -> List[Chunk]:
        chunks = []
        for file in files:
            content = AbstractChunker.read_file(file)
            if content.strip():
                file_chunks = self.chunk_file(file, content)
                chunks.extend(file_chunks)
        return chunks

    @abstractmethod
    def chunk_file(self,
                   file_name: Path,
                   content: str) -> List[Chunk]:
        pass

    @abstractmethod
    def make_chunk(self, text: str, source: str, first_char_idx: int,
                   last_char_idx: int, original_chunk_id: str | None,
                   *args, **kwargs) -> Chunk:
        pass

    def enforce_char_limit(self, text: str) -> List[str]:
        """Hard-split a chunk that still exceeds max_chunk_size, on line boundaries."""
        if len(text) <= self.max_chunk_size:
            return [text]
        lines = text.splitlines(keepends=True)
        sub_chunks, current, current_len = [], [], 0
        for line in lines:
            line_len = len(line)
            if current_len + line_len > self.max_chunk_size and current:
                sub_chunks.append("".join(current))
                current, current_len = [], 0
            current.append(line)
            current_len += line_len
        if current:
            sub_chunks.append("".join(current))
        return sub_chunks
