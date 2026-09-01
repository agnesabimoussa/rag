from pathlib import Path
import json
import pickle
from rank_bm25 import BM25Okapi
from typing import List
from typing import Any
from src.error_handling.inavlid_json import InvalidJSON
from pydantic import TypeAdapter, ValidationError


class FileOperations:
    @staticmethod
    def resolve_files(path: Path, suffix: str) -> List[Path]:
        if path.is_file():
            return [path]
        if path.is_dir():
            files = sorted(
                file for file in path.iterdir()
                if file.is_file() and file.suffix.lower() == suffix.lower()
            )
            if not files:
                raise FileNotFoundError(f"No {suffix} files found under {path}")
            return files
        raise FileNotFoundError(f"Path does not exist: {path}")

    @staticmethod
    def load_content(file: Path, type: Any) -> Any:
        try:
            with open(file, "r", encoding="utf-8") as opened:
                content = json.load(opened)
                return TypeAdapter(type).validate_python(content)
        except (ValidationError, json.JSONDecodeError, OSError):
            raise InvalidJSON("InvalidJSON exception occured."
                              f"{file} contains invalid JSON.")

    @staticmethod
    def write_json(path: Path, content: Any):
        with open(path, "w", encoding="utf-8") as opened:
            json.dump(content.model_dump(),
                      opened,
                      indent=4)

    @staticmethod
    def read_file(file: Path) -> str:
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
        return content

    @staticmethod
    def load_bm25(file: Path) -> BM25Okapi:
        with open(file, "rb") as f:
            bm25 = pickle.load(f)
        return bm25

    @staticmethod
    def write_bm25(file: Path, bm25: BM25Okapi) -> None:
        with open(file, "wb") as f:
            pickle.dump(bm25, f)
