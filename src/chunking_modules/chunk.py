from pydantic import BaseModel


class Chunk(BaseModel):
    """A single indexed piece of text, with its source location."""

    text: str
    source: str
    chunk_id: int
    first_character_index: int
    last_character_index: int
