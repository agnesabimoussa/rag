from pathlib import Path
import json
from typing import List
from typing import Any
from src.error_handling_modules.inavlid_json import InvalidJSON
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
