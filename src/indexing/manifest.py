"""Tracks each source file's content hash and the chunk IDs it produced.

Comparing a fresh scan of the corpus against this manifest is what lets
`ChunksGenerator` figure out which files are unchanged (skip), changed
(re-chunk and re-embed), or deleted (drop from every index) without
re-processing the whole corpus on every run.
"""
from pathlib import Path
from typing import cast, Dict, List

from pydantic import BaseModel

from src.error_handling.inavlid_json import InvalidJSON
from src.utils.file_operations import FileOperations


class FileManifestEntry(BaseModel):
    hash: str
    chunk_ids: List[str] = []


class IndexManifest(BaseModel):
    files: Dict[str, FileManifestEntry] = {}

    @classmethod
    def load(cls, path: Path) -> "IndexManifest":
        if not path.is_file():
            return cls()
        try:
            return cast(IndexManifest, FileOperations.load_content(path, IndexManifest))
        except InvalidJSON:
            return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        FileOperations.write_json(path, self)
