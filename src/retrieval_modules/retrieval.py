from src.data_models.unanswered_question import UnansweredQuestion
from src.data_models.answered_question import AnsweredQuestion
from src.error_handling_modules.inavlid_json import InvalidJSON
from rank_bm25 import BM25Okapi
from pathlib import Path
from typing import List
import json
from pydantic import TypeAdapter, ValidationError


class Retrieval:
    def __init__(self, bm25: BM25Okapi, folder_path: str, k: int = 4):
        self.bm25 = bm25
        self.prompts = self._read_prompts(folder_path)
        self.k = k

    @staticmethod
    def _read_prompts(folder_path: str) -> List[UnansweredQuestion]:
        prompts = []
        folder = Path(folder_path)
        adapter = TypeAdapter(list[UnansweredQuestion])
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Input directory does not exist: {folder}"
            )
        for file_path in folder.rglob("*"):
            if file_path.is_file() and file_path.suffix == ".json":
                with file_path.open("r", encoding="utf-8") as file:
                    content = json.load(file)
                try:
                    questions = adapter.validate_python(
                        content["rag_questions"])
                    prompts.extend(questions)
                except ValidationError:
                    raise InvalidJSON("InvalidJSON exception occured.")
        return prompts

    def answer_prompts(self) -> List[AnsweredQuestion]:
        answers = []

    def get_prompts(self) -> None:
        for prompt in self.prompts:
            print(prompt.question)
