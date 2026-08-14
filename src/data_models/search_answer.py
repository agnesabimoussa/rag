from data_models.search_result import MinimalSearchResults
from typing import List
from pydantic import BaseModel


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResultsAndAnswer(BaseModel):
    search_results: List[MinimalAnswer]
    k: int
