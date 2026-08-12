from pydantic import BaseModel, Field
import uuid


class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda:
                             str(uuid.uuid4()))
    question: str
