from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
from pydantic import TypeAdapter, ValidationError
from src.data_models.search_result import StudentSearchResults, MinimalSearchResults
from src.data_models.rag_dataset import RagDataset
from src.error_handling_modules.inavlid_json import InvalidJSON
from src.error_handling_modules.invalid_test import InvalidTest
from src.data_models.minimal_source import MinimalSource
from src.data_models.answered_question import AnsweredQuestion
from src.data_models.unanswered_question import UnansweredQuestion


# recall@k = top k relevant chunks retrieved by the retrieval system / total relevant ground truth chunks
class Evaluation:
    def __init__(self,
                 student_search_results_path: str,
                 dataset_path: str) -> None:
        self.__student_search_results_path = Path(student_search_results_path)
        self.__dataset_path = Path(dataset_path)
        self.__student_results: StudentSearchResults = Evaluation.__load_content(
            self.__student_search_results_path, StudentSearchResults)
        self.__ground_truth: RagDataset = Evaluation.__load_content(self.__dataset_path, RagDataset)
        self.__validate()

    @staticmethod
    def __load_content(file: Path, type: Any) -> Any:
        try:
            with open(file, "r", encoding="utf-8") as file:
                content = json.load(file)
                return TypeAdapter(type).validate_python(content)
        except (ValidationError, json.JSONDecodeError, OSError):
            raise InvalidJSON("InvalidJSON exception occured."
                              f"{file} contains invalid JSON.")

    def __validate(self) -> None:
        if len(self.__student_results.search_results) != len(self.__ground_truth.rag_questions):
            raise InvalidTest("InvalidTest: student results file and ground truth file"
                              "should have the same number of questions.")

    @staticmethod
    def __calculate_iou(
        retrieved_start: int,
        retrieved_end: int,
        ground_start: int,
        ground_end: int
    ) -> float:
        intersection = max(
            0,
            min(retrieved_end, ground_end) - max(retrieved_start, ground_start)
        )
        union = max(retrieved_end, ground_end) - min(retrieved_start, ground_start)
        return intersection / union

    def evaluate(self, k: int = 5) -> float:
        if k <= 0:
            k = 5

        search_results: List[MinimalSearchResults] = self.__student_results.search_results
        search_answers: List[AnsweredQuestion | UnansweredQuestion] = self.__ground_truth.rag_questions

        size = len(search_results)
        total_recall = 0.0

        for i in range(size):
            if search_answers[i].question != search_results[i].question:
                raise InvalidTest(
                    "InvalidTest: student results file and ground truth file "
                    "should contain the same questions in the same order."
                )

            student_sources = search_results[i].retrieved_sources[:k]
            ground_truth_sources = search_answers[i].sources

            if not ground_truth_sources:
                continue

            correct_sources = 0

            for source in ground_truth_sources:
                for student_source in student_sources:
                    if (
                        student_source.file_path == source.file_path
                        and self.__calculate_iou(
                            student_source.first_character_index,
                            student_source.last_character_index,
                            source.first_character_index,
                            source.last_character_index
                        ) >= 0.05
                    ):
                        correct_sources += 1
                        break

            question_recall = correct_sources / len(ground_truth_sources)
            total_recall += question_recall

        return total_recall / size if size > 0 else 0.0

    def print_report(self) -> None:
        pass
