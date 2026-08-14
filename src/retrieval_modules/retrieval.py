from src.data_models.unanswered_question import UnansweredQuestion
from rank_bm25 import BM25Okapi
from pathlib import Path
from typing import List


class Retrieval:
    def __init__(self, bm25: BM25Okapi, input_file: str, k: int = 4):
        self.bm25 = bm25
        self.prompts = _read_prompts(input_file)
        self.k = k

    def _read_prompts(input_file: str) -> List[UnansweredQuestion]:
        pass
