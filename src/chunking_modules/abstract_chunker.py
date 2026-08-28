import itertools
from typing import List
from pathlib import Path
from abc import ABC, abstractmethod
from src.chunking_modules.chunk import Chunk


class AbstractChunker(ABC):
    def __init__(self, max_chunk_size: int) -> None:
        self.id_generator = itertools.count(start=1)
        self.max_chunk_size = max_chunk_size

    @staticmethod
    def find_span(text: str, target: str, cursor: int = 0) -> tuple[int, int]:
        """Find a target's source span, tolerating splitter-only whitespace changes.

        ``MarkdownHeaderTextSplitter`` removes indentation from some Markdown
        blocks (notably HTML blocks).  Match exactly where possible, then fall
        back to a whitespace-insensitive match while returning offsets in the
        original source text.
        """
        start = text.find(target, cursor)
        if start == -1:
            start = text.find(target)
        if start != -1:
            return start, start + len(target)

        normalized_text = []
        source_positions = []
        for index, char in enumerate(text):
            if not char.isspace():
                normalized_text.append(char)
                source_positions.append(index)
        normalized_target = "".join(char for char in target if not char.isspace())
        if not normalized_target:
            raise ValueError(f"Could not locate non-whitespace text: {target[:80]!r}")

        normalized_source = "".join(normalized_text)
        normalized_cursor = next(
            (index for index, position in enumerate(source_positions) if position >= cursor),
            len(normalized_source),
        )
        normalized_start = normalized_source.find(normalized_target, normalized_cursor)
        if normalized_start == -1:
            normalized_start = normalized_source.find(normalized_target)
        if normalized_start == -1:
            raise ValueError(f"Could not locate text: {target[:80]!r}")
        normalized_end = normalized_start + len(normalized_target) - 1
        return source_positions[normalized_start], source_positions[normalized_end] + 1

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
