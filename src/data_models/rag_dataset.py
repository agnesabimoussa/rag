from pydantic import BaseModel
from data_models.answered_question import AnsweredQuestion
from data_models.unanswered_question import UnansweredQuestion
from typing import List


class RagDataset(BaseModel):
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]
