from src.data_models.search_result import StudentSearchResults
from src.data_models.search_answer import StudentSearchResultsAndAnswer
from src.error_handling_modules.inavlid_json import InvalidJSON
from pathlib import Path
from pydantic import TypeAdapter, ValidationError
from typing import List
import json

class AnswerGenerator:
    def __init__(self,
                 student_results_path: str,
                 save_dir: str) -> None:
        self.student_results_path = Path(student_results_path)
        self.save_dir = Path(save_dir)
        self.search_results = self._read_search_results()

    def _read_search_results(self) -> StudentSearchResults:
        try:
            adapter = TypeAdapter(List[StudentSearchResults])
            with open(self.student_results_path, "r", encoding="utf-8") as file:
                content = json.load(file)
            search_results = adapter.validate_python(content)
            return search_results
        except ValidationError:
            raise InvalidJSON("InvalidJSON exception occured."
                              f"{self.student_results_path} contains invalid JSON.")

    def answer_dataset(self) -> StudentSearchResultsAndAnswer:
        for prompt in self.search_results:
            