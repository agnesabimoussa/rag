# from markdown_indexing import MarkdwonIndexing
# from code_indexing import CodeIndexing
from pathlib import Path
import bisect
import itertools
import json
from typing import List, Tuple
from src.chunking_modules.markdown_chunking import MarkdwonChunking
from src.chunking_modules.code_chunking import CodeChunking
from src.chunking_modules.chunk import Chunk


def _normalize_with_positions(text: str) -> Tuple[str, List[int]]:
    """Strip everything but word characters, remembering each kept
    character's original index so a match in the normalized string can be
    mapped back to a real offset."""
    normalized_chars = []
    positions = []
    for index, char in enumerate(text):
        if char.isalnum() or char == "_":
            normalized_chars.append(char)
            positions.append(index)
    return "".join(normalized_chars), positions


class _SpanLocator:
    """Locates each chunk's (first_character_index, last_character_index)
    span in its source content, tolerant of splitters (e.g.
    MarkdownHeaderTextSplitter) that reformat or drop whitespace/punctuation
    while leaving word content untouched."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.norm_content, self.positions = _normalize_with_positions(content)
        self.search_start = 0

    def locate(self, chunk: str) -> Tuple[int, int]:
        # self.search_start is always in raw-content coordinates; the
        # normalized string is only searched via a position translated
        # through `self.positions` so the two coordinate spaces never mix.
        exact_index = self.content.find(chunk, self.search_start)
        if exact_index != -1:
            self.search_start = exact_index
            return exact_index, exact_index + len(chunk)

        norm_chunk, _ = _normalize_with_positions(chunk)
        if not norm_chunk:
            raise ValueError(f"Chunk text not found in source content: {chunk[:80]!r}")
        norm_search_start = bisect.bisect_left(self.positions, self.search_start)
        norm_start = self.norm_content.find(norm_chunk, norm_search_start)
        if norm_start == -1:
            raise ValueError(f"Chunk text not found in source content: {chunk[:80]!r}")
        norm_end = norm_start + len(norm_chunk) - 1
        first_character_index = self.positions[norm_start]
        last_character_index = self.positions[norm_end] + 1
        self.search_start = first_character_index
        return first_character_index, last_character_index


class Chunking:
    def __init__(self, folder_path: str, output_file_path: str,
                 max_chunk_size: int = 2000) -> None:
        self.folder_path = Path(folder_path)
        self.output_file_path = Path(output_file_path)
        self.max_chunk_size = max_chunk_size
        self.mardown_chunking = MarkdwonChunking(max_chunk_size)
        self.code_chunking = CodeChunking(max_chunk_size)
        self.chunks: list[Chunk] = []
        self.id_generator = itertools.count(start=1)

    def _add_chunks(self, chunks: List[str], source: str, content: str):
        locator = _SpanLocator(content)
        for chunk in chunks:
            first_character_index, last_character_index = locator.locate(chunk)
            self.chunks.append(Chunk(
                text=chunk,
                source=source,
                chunk_id=next(self.id_generator),
                first_character_index=first_character_index,
                last_character_index=last_character_index
            ))

    def chunk_files(self) -> None:
        if not self.folder_path.is_dir():
            raise FileNotFoundError(
                f"Input directory does not exist: {self.folder_path}"
            )

        for file_path in self.folder_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() == ".md":
                with file_path.open("r", encoding="utf-8") as file:
                    content = file.read()
                chunks = self.mardown_chunking.chunk_file(content)
                source = file.name
                self._add_chunks(chunks, source, content)
            elif file_path.is_file() and file_path.suffix.lower() == ".py":
                with file_path.open("r", encoding="utf-8") as file:
                    content = file.read()
                chunks = self.code_chunking.chunk_file(content)
                source = file.name
                self._add_chunks(chunks, source, content)

    def get_chunks(self) -> List[Chunk]:
        return self.chunks

    def write_result(self) -> None:
        self.chunk_files()
        output_path = self.output_file_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                [chunk.model_dump() for chunk in self.chunks],
                file,
                indent=4
            )
