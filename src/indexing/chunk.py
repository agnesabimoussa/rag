from dataclasses import asdict, dataclass


@dataclass
class Chunk:
    chunk_id: int
    chunk_text: str

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)
