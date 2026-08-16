from pydantic import BaseModel, Field
import uuid


class UnansweredQuestion(BaseModel):
    """A question to be answered by the RAG pipeline, with no ground truth."""

    question_id: str = Field(default_factory=lambda:
                             str(uuid.uuid4()))
    question: str
