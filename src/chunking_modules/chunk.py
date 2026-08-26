from pydantic import BaseModel, Field
from typing import List


class Chunk(BaseModel):
    id: str
    text: str
    source: str
    first_character_index: int
    last_character_index: int
    tokens: int
    original_chunk_id: str | None = Field(None)


class MarkdownChunk(Chunk):
    section: str


class CodeChunk(Chunk):
    # function, docstring, async function, method, data member
    type: str | None = Field(None)
    # direct parent
    parent_id: str | None = Field(None)
    # used for nested functions / classes
    child_ids: List[str] | None = Field(None)
