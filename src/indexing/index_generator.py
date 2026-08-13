# from markdown_indexing import MarkdwonIndexing
# from code_indexing import CodeIndexing
from pathlib import Path
from src.indexing.markdown_indexing import MarkdwonIndexing
from src.indexing.code_indexing import CodeIndexing
from typing import Dict
import itertools
import json


class Indexing:
    def __init__(self, folder_path: str, output_file_path: str,
                 max_chunk_size: int | None = None) -> None:
        self.folder_path = folder_path
        self.output_file_path = output_file_path
        self.max_chunk_size = max_chunk_size
        self.mardown_indexing = MarkdwonIndexing(max_chunk_size)
        self.code_indexing = CodeIndexing(max_chunk_size)

    def index_files(self) -> Dict[int, str]:
        all_chunks = []
        id_chunk = {}
        for file_path in Path(self.folder_path).rglob("*"):
            if file_path.is_file() and file_path.name.endswith(".md"):
                content = file_path.open("r", encoding="utf-8").read()
                chunks = self.mardown_indexing.index_file(content)
                for chunk in chunks:
                    all_chunks.append(chunk)
            if file_path.is_file() and file_path.name.endswith(".py"):
                content = file_path.open("r", encoding="utf-8").read()
                chunks = self.code_indexing.index_file(content)
                for chunk in chunks:
                    all_chunks.append(chunk)
        id_generator = itertools.count(start=1)
        for chunk in all_chunks:
            id = next(id_generator)
            id_chunk[id] = chunk
        return id_chunk

    def write_result(self) -> None:
        id_chunk = self.index_files()
        folder_path = Path("/data/processed/")
        file_name = "index_file.json"
        full_path = folder_path / file_name
        folder_path.mkdir(parents=True, exist_ok=True)
        with open(full_path, "w") as file:
            json.dump(id_chunk, file, indent=4, sort_keys=True)
