# from markdown_indexing import MarkdwonIndexing
# from code_indexing import CodeIndexing
from pathlib import Path
import itertools
import json
from typing import List
from src.chunking_modules.markdown_chunking import MarkdwonChunking
from src.chunking_modules.code_chunking import CodeChunking
from src.chunking_modules.chunk import Chunk


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

    def _add_chunks(self, chunks: List[str], source: str):
        for chunk in chunks:
            self.chunks.append(Chunk(
                text=chunk,
                source=source,
                chunk_id=next(self.id_generator)
            ))

    def chunk_files(self) -> list[Chunk]:
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
                self._add_chunks(chunks, source)
            elif file_path.is_file() and file_path.suffix.lower() == ".py":
                with file_path.open("r", encoding="utf-8") as file:
                    content = file.read()
                chunks = self.code_chunking.chunk_file(content)
                source = file.name
                self._add_chunks(chunks, source)

    # def print_chunks(self):
    #     for chunk in self.chunks:
    #         print(chunk)
    
    def get_chunks_text(self) -> List[str]:
        chunks_text = []
        for chunk in self.chunks:
            chunks_text.append(chunk.text)
        return chunks_text

    def write_result(self) -> None:
        output_path = self.output_file_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                [chunk.model_dump() for chunk in self.chunks],
                file,
                indent=4
            )
