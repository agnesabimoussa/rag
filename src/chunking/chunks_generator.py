"""Chunk generation and source-span mapping for the document corpus."""
from pathlib import Path
from typing import List
from tqdm import tqdm
from src.chunking.code_chunker import CodeChunker
from src.chunking.markdown_chunker import MarkdownChunker
from src.data_models.chunk import Chunk
from src.ingestion.file_operations import FileOperations


class ChunksGenerator:
    """Splits every Markdown/Python file under a folder into `Chunk`s,
    using a distinct chunking strategy per file type.
    """

    def __init__(self, corpus_path: str = "data/raw/",
                 output_dir: str = "data/processed/",
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
        self.markdown_chunking_strategy = MarkdownChunker(max_chunk_size)
        self.code_chunking_strategy = CodeChunker(max_chunk_size)

    def get_files(self) -> None:
        if not self.corpus_path.is_dir():
            raise FileNotFoundError(
                f"Input directory does not exist: {self.corpus_path}"
            )

        file_paths = sorted(self.corpus_path.rglob("*"))
        for file_path in tqdm(file_paths, desc="Chunking"):
            if file_path.is_file() and file_path.suffix.lower() == ".md":
                self.markdown_files.append(file_path)
            elif file_path.is_file() and file_path.suffix.lower() == ".py":
                self.python_files.append(file_path)

    @staticmethod
    def write_result(path: Path, file_name: str, content: List[Chunk]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        full_path = path / file_name
        FileOperations.write_json(full_path, content)

    def apply_chunking(self) -> List[Chunk]:
        markdown_chunks = self.markdown_chunking_strategy.chunk_files(self.markdown_files)
        code_chunks = self.code_chunking_strategy.chunk_files(self.python_files)
        self.write_result(self.output_dir, "markdown_chunks.json", markdown_chunks)
        self.write_result(self.output_dir, "code_chunks.json", code_chunks)
        return markdown_chunks + code_chunks
