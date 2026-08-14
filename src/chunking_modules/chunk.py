from pydantic import BaseModel

class Chunk(BaseModel):
    text: str
    source: str
    chunk_id: int