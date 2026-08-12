from unanswered_question import UnansweredQuestion
from minimal_source import MinimalSource
from typing import List


class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str
