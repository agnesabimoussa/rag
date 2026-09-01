from src.data_models.search_result import (StudentSearchResults,
                                           MinimalSearchResults)
from src.data_models.search_answer import (StudentSearchResultsAndAnswer,
                                           MinimalAnswer)
from src.data_models.minimal_source import MinimalSource
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from src.models.llm import LLM
from src.utils.file_operations import FileOperations


class AnswerGenerator:
    def __init__(self,
                 student_search_results_path: str = "data/output/search_results/",
                 save_dir: str = "data/output/search_results_and_answer/") -> None:
        self.student_search_results_path = Path(student_search_results_path)
        self.save_dir = Path(save_dir)
        self.model = LLM()
        self.cache: Dict[str, str] = {}

    def _get_retrieved_context(self, sources: List[MinimalSource]) -> str:
        context = []
        for source in sources:
            content = FileOperations.read_file(Path(source.file_path))
            retrieved_text = content[
                source.first_character_index:source.last_character_index
            ]
            label = f"# Source: {source.file_path}"
            if source.scope:
                label += f" ({source.scope})"
            context.append(f"{label}\n{retrieved_text}")
        return "\n\n".join(context)

    def answer_prompt(self, question: MinimalSearchResults) -> str:
        # cache hit
        if question.question_id in self.cache:
            return self.cache[question.question_id]
        # else if cache miss
        messages: List[Dict[str, str]] = []
        context = self._get_retrieved_context(
            question.retrieved_sources
        )
        self.model.add_user_message(messages, context)
        self.model.add_user_message(messages, question.question)
        response = self.model.chat(messages)
        self.cache[question.question_id] = response
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
            search_results = FileOperations.load_content(file, StudentSearchResults)
            answers = self.answer_dataset(search_results)
            full_path = self.save_dir / file.name
            FileOperations.write_json(full_path, answers)
