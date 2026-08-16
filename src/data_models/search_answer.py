from src.data_models.search_result import MinimalSearchResults
from typing import List
from pydantic import BaseModel


class MinimalAnswer(MinimalSearchResults):
    """A question, its retrieved sources, and the generated answer."""

    answer: str


class StudentSearchResultsAndAnswer(BaseModel):
    """Search results with answers for a whole dataset, as written by
    `answer_dataset`."""

    search_results: List[MinimalAnswer]
    k: int
