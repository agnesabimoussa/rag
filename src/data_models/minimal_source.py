from pydantic import BaseModel
from typing import Optional


class MinimalSource(BaseModel):
    """A single retrieved source: a corpus-relative file path and the
    character range within it, compared verbatim against ground truth."""

    file_path: str
    first_character_index: int
    last_character_index: int
    scope: Optional[str] = None
