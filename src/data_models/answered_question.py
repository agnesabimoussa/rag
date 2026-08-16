from src.data_models.unanswered_question import UnansweredQuestion
from src.data_models.minimal_source import MinimalSource
from typing import List


class AnsweredQuestion(UnansweredQuestion):
    """A question with its ground-truth sources and answer, used to
    evaluate retrieval and generation quality."""

    sources: List[MinimalSource]
    answer: str
