from src.data_models.search_result import (StudentSearchResults,
                                           MinimalSearchResults)
from src.data_models.search_answer import (StudentSearchResultsAndAnswer,
                                           MinimalAnswer)
from src.data_models.minimal_source import MinimalSource
from src.error_handling_modules.inavlid_json import InvalidJSON
from pathlib import Path
from pydantic import TypeAdapter, ValidationError
from typing import List
import json
from tqdm import tqdm
from src.models.language_model import LLM


class AnswerGenerator:
    def __init__(self,
                 student_search_results_path: str,
                 save_dir: str) -> None:
        self.student_search_results_path = Path(student_search_results_path)
        self.save_dir = Path(save_dir)
        self.search_results = self._read_search_results()
        self.k = self.search_results.k
        self.model = LLM()

    def _read_search_results(self) -> StudentSearchResults:
        try:
            adapter = TypeAdapter(StudentSearchResults)
            with open(self.student_search_results_path, "r", encoding="utf-8") as file:
                content = json.load(file)
            return adapter.validate_python(content)
        except (ValidationError, json.JSONDecodeError):
            raise InvalidJSON("InvalidJSON exception occured."
                              f"{self.student_search_results_path} contains invalid JSON.")

    def _get_retrieved_context(self, sources: List[MinimalSource]) -> str:
        context = []
        for source in sources:
            with open(source.file_path, "r", encoding="utf-8") as file:
                content = file.read()
            retrieved_text = content[
                source.first_character_index:source.last_character_index
            ]
            context.append(retrieved_text)
        return "\n\n".join(context)

    def answer_prompt(self, question: MinimalSearchResults) -> str:
        messages = []
        context = self._get_retrieved_context(
            question.retrieved_sources
        )
        self.model.add_user_message(messages, context)
        self.model.add_user_message(messages, question.question)
        response = self.model.chat(messages)
        return response

    def answer_dataset(self) -> StudentSearchResultsAndAnswer:
        search_results_and_answer = StudentSearchResultsAndAnswer(k=self.k, search_results=[])
        for search_result in tqdm(self.search_results.search_results, desc="Generating answers"):
            answer = MinimalAnswer(**search_result.model_dump(),
                                   answer=self.answer_prompt(search_result))
            search_results_and_answer.search_results.append(answer)
        return search_results_and_answer

    def write_answers(self) -> None:
        answers = self.answer_dataset()
        full_path = self.save_dir / self.student_search_results_path.name
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as file:
            json.dump(answers.model_dump(),
                      file,
                      indent=4)
