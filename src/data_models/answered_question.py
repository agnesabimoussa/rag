from data_models.unanswered_question import UnansweredQuestion
from data_models.minimal_source import MinimalSource
from typing import List


class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str
