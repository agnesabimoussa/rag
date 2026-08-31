from pathlib import Path
from typing import List


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
    
    
