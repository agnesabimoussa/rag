from pydantic import BaseModel
from typing import List
from minimal_source import MinimalSource


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int
