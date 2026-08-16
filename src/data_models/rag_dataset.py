from pydantic import BaseModel
from src.data_models.answered_question import AnsweredQuestion
from src.data_models.unanswered_question import UnansweredQuestion
from typing import List


class RagDataset(BaseModel):
    """A dataset of RAG questions, answered or not."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]
