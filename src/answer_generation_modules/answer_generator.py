from src.data_models.search_result import (StudentSearchResults,
                                           MinimalSearchResults)
from src.data_models.search_answer import (StudentSearchResultsAndAnswer,
                                           MinimalAnswer)
from src.data_models.minimal_source import MinimalSource
from src.error_handling_modules.inavlid_json import InvalidJSON
from src.file_operations.file_operations import FileOperations
from pathlib import Path
from pydantic import TypeAdapter, ValidationError
from typing import List
import json
from tqdm import tqdm
from src.models.language_model import LLM


class AnswerGenerator:
    def __init__(self,
                 student_search_results_path: str = "data/output/search_results/",
                 save_dir: str = "data/output/search_results_and_answer/") -> None:
        self.student_search_results_path = Path(student_search_results_path)
        self.save_dir = Path(save_dir)
        self.model = LLM()

    def _read_search_results(self, file: Path) -> StudentSearchResults:
        try:
            adapter = TypeAdapter(StudentSearchResults)
            with open(file, "r", encoding="utf-8") as opened:
                content = json.load(opened)
            return adapter.validate_python(content)
        except (ValidationError, json.JSONDecodeError):
            raise InvalidJSON("InvalidJSON exception occured."
                              f"{file} contains invalid JSON.")

    def _get_retrieved_context(self, sources: List[MinimalSource]) -> str:
        context = []
        for source in sources:
            with open(source.file_path, "r", encoding="utf-8") as file:
                content = file.read()
            retrieved_text = content[
                source.first_character_index:source.last_character_index
            ]
            label = f"# Source: {source.file_path}"
            if source.scope:
                label += f" ({source.scope})"
            context.append(f"{label}\n{retrieved_text}")
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

    def answer_dataset(self, search_results: StudentSearchResults) -> StudentSearchResultsAndAnswer:
        search_results_and_answer = StudentSearchResultsAndAnswer(k=search_results.k, search_results=[])
        for search_result in tqdm(search_results.search_results, desc="Generating answers"):
            answer = MinimalAnswer(**search_result.model_dump(),
                                   answer=self.answer_prompt(search_result))
            search_results_and_answer.search_results.append(answer)
        return search_results_and_answer

    def write_answers(self) -> None:
        files = FileOperations.resolve_files(self.student_search_results_path, ".json")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        for file in files:
            search_results = self._read_search_results(file)
            answers = self.answer_dataset(search_results)
            full_path = self.save_dir / file.name
            with open(full_path, "w", encoding="utf-8") as opened:
                json.dump(answers.model_dump(),
                          opened,
                          indent=4)
