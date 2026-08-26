"""Chunk generation and source-span mapping for the document corpus."""

import bisect
import json
import itertools
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
from src.chunking_modules.code_chunking import CodeChunking
from src.chunking_modules.markdown_chunking import MarkdwonChunking
from src.chunking_modules.chunk import (MarkdownChunk, CodeChunk)


def _normalize_with_positions(text: str) -> Tuple[str, List[int]]:
    """Strip everything but word characters, remembering each kept
    character's original index so a match in the normalized string can be
    mapped back to a real offset.

    Args:
        text: The input text to normalize.

    Returns:
        A tuple containing the normalized text and the original positions of
        each retained character.
    """
    normalized_chars = []
    positions = []
    for index, char in enumerate(text):
        if char.isalnum() or char == "_":
            normalized_chars.append(char)
            positions.append(index)
    return "".join(normalized_chars), positions


class _SpanLocator:
    """Locate each chunk span in its original source content."""

    def __init__(self, content: str) -> None:
        """Prepare a locator for a source document.

        Args:
            content: The complete raw source content.
        """
        self.content = content
        self.norm_content, self.positions = _normalize_with_positions(content)
        self.search_start = 0

    def locate(self, chunk: str) -> Tuple[int, int]:
        """Find the original character offsets for a chunk.

        Args:
            chunk: The chunk text as produced by the splitter.

        Returns:
            A tuple of ``(first_character_index, last_character_index)`` values.

        Raises:
            ValueError: If the chunk cannot be mapped back to the source.
        """
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
    """Splits every Markdown/Python file under a folder into `Chunk`s,
    using a distinct chunking strategy per file type.
    """

    def __init__(self, corpus_path: str, output_dir: str,
                 max_chunk_size: int = 2000) -> None:
        """Initialize the chunker.

        Args:
            corpus_path: Root directory to recursively scan for `.md`/`.py`
                files.
            output_file_path: Where `write_result` persists the chunks.
            max_chunk_size: Maximum characters per chunk.
        """
        self.corpus_path = Path(corpus_path)
        self.output_dir = Path(output_dir)
        self.markdown_files: List[Path] = []
        self.python_files: List[Path] = []
        self.get_files()
        self.markdown_chunking_strategy = MarkdwonChunking(max_chunk_size)
        self.code_chunking_strategy = CodeChunking(max_chunk_size)

    def get_files(self) -> None:
        if not self.corpus_path.is_dir():
            raise FileNotFoundError(
                f"Input directory does not exist: {self.corpus_path}"
            )

        file_paths = list(self.corpus_path.rglob("*"))
        for file_path in tqdm(file_paths, desc="Chunking"):
            if file_path.is_file() and file_path.suffix.lower() == ".md":
                self.markdown_files.append(file_path)
            elif file_path.is_file() and file_path.suffix.lower() == ".py":
                self.python_files.append(file_path)

    @staticmethod
    def write_result(path: Path, file_name: str, content: List[MarkdownChunk] | List[CodeChunk]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        full_path = path / file_name
        with full_path.open("w", encoding="utf-8") as file:
            json.dump(
                [chunk.model_dump() for chunk in content],
                file,
                indent=4
            )

    def apply_chunking(self) -> List[MarkdownChunk | CodeChunk]:
        markdown_chunks = self.markdown_chunking_strategy.chunk_files(self.markdown_files)
        code_chunks = self.code_chunking_strategy.chunk_files(self.python_files)
        self.write_result(self.output_dir, "markdown_chunks.json", markdown_chunks)
        self.write_result(self.output_dir, "code_chunks.json", code_chunks)
        return markdown_chunks + code_chunks
