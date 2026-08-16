from pydantic import BaseModel


class Chunk(BaseModel):
    text: str
    source: str
    chunk_id: int
    first_character_index: int
    last_character_index: int
