from pydantic import BaseModel
from answered_question import AnsweredQuestion
from unanswered_question import UnansweredQuestion
from typing import List


class RagDataset(BaseModel):
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]
