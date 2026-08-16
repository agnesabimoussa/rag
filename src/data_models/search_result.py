from pydantic import BaseModel
from typing import List
from src.data_models.minimal_source import MinimalSource


class MinimalSearchResults(BaseModel):
    """A question and the sources retrieved for it."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class StudentSearchResults(BaseModel):
    """Search results for a whole dataset, as written by `search_dataset`."""

    search_results: List[MinimalSearchResults]
    k: int
