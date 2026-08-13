# from markdown_indexing import MarkdwonIndexing
# from code_indexing import CodeIndexing
from pathlib import Path
from src.indexing.markdown_indexing import MarkdwonIndexing
from src.indexing.code_indexing import CodeIndexing
from src.indexing.chunk import Chunk
import itertools
import json


class Indexing:
    def __init__(self, folder_path: str, output_file_path: str,
                 max_chunk_size: int = 1000) -> None:
        self.folder_path = Path(folder_path)
        self.output_file_path = Path(output_file_path)
        self.max_chunk_size = max_chunk_size
        self.mardown_indexing = MarkdwonIndexing(max_chunk_size)
        self.code_indexing = CodeIndexing(max_chunk_size)

    def index_files(self) -> list[Chunk]:
        if not self.folder_path.is_dir():
            raise FileNotFoundError(
                f"Input directory does not exist: {self.folder_path}"
            )

        all_chunks: list[str] = []
        for file_path in self.folder_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() == ".md":
                with file_path.open("r", encoding="utf-8") as file:
                    content = file.read()
                chunks = self.mardown_indexing.index_file(content)
                all_chunks.extend(chunks)
            elif file_path.is_file() and file_path.suffix.lower() == ".py":
                with file_path.open("r", encoding="utf-8") as file:
                    content = file.read()
                chunks = self.code_indexing.index_file(content)
                all_chunks.extend(chunks)
        id_generator = itertools.count(start=1)
        return [Chunk(next(id_generator), chunk) for chunk in all_chunks]

    def write_result(self) -> None:
        id_chunk = self.index_files()
        output_path = self.output_file_path
        if output_path.suffix.lower() != ".json":
            output_path /= "index_file.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump([chunk.to_dict() for chunk in id_chunk], file, indent=4)
