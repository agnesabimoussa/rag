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
from src.file_operations.file_operations import FileOperations


# recall@k = top k relevant chunks retrieved by the retrieval system / total relevant ground truth chunks
class Evaluation:
    def __init__(self,
                 student_search_results_path: str = "data/output/search_results/",
                 dataset_path: str = "data/datasets/AnsweredQuestions/") -> None:
        self.__student_search_results_path = Path(student_search_results_path)
        self.__dataset_path = Path(dataset_path)
        self.__cases: List[Tuple[str, StudentSearchResults, RagDataset]] = Evaluation.__load_cases(
            self.__student_search_results_path, self.__dataset_path)
        self.__validate()

    @staticmethod
    def __load_content(file: Path, type: Any) -> Any:
        try:
            with open(file, "r", encoding="utf-8") as opened:
                content = json.load(opened)
                return TypeAdapter(type).validate_python(content)
        except (ValidationError, json.JSONDecodeError, OSError):
            raise InvalidJSON("InvalidJSON exception occured."
                              f"{file} contains invalid JSON.")

    @staticmethod
    def __load_cases(student_search_results_path: Path,
                     dataset_path: Path) -> List[Tuple[str, StudentSearchResults, RagDataset]]:
        student_files = FileOperations.resolve_files(student_search_results_path, ".json")
        ground_truth_by_name = {
            file.name: file for file in FileOperations.resolve_files(dataset_path, ".json")
        }
        cases = []
        for student_file in student_files:
            ground_truth_file = ground_truth_by_name.get(student_file.name)
            if ground_truth_file is None:
                raise InvalidTest(
                    "InvalidTest: no ground truth file found matching "
                    f"{student_file.name} under {dataset_path}."
                )
            student_results = Evaluation.__load_content(student_file, StudentSearchResults)
            ground_truth = Evaluation.__load_content(ground_truth_file, RagDataset)
            cases.append((student_file.name, student_results, ground_truth))
        return cases

    def __validate(self) -> None:
        for name, student_results, ground_truth in self.__cases:
            if len(student_results.search_results) != len(ground_truth.rag_questions):
                raise InvalidTest(
                    f"InvalidTest: student results file and ground truth file for {name} "
                    "should have the same number of questions."
                )

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

        size = 0
        total_recall = 0.0

        for name, student_results, ground_truth in self.__cases:
            search_results: List[MinimalSearchResults] = student_results.search_results
            search_answers: List[AnsweredQuestion | UnansweredQuestion] = ground_truth.rag_questions
            size += len(search_results)

            for i in range(len(search_results)):
                if search_answers[i].question != search_results[i].question:
                    raise InvalidTest(
                        f"InvalidTest: student results file and ground truth file for {name} "
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
        recall1 = self.evaluate(1)
        recall3 = self.evaluate(3)
        recall5 = self.evaluate(5)
        recall10 = self.evaluate(10)

        total_questions = sum(len(student_results.search_results)
                              for _, student_results, _ in self.__cases)

        print("\nEvaluation Results")
        print("=" * 60)
        print(f"Questions evaluated: {total_questions}")
        print(f"Recall@1:  {recall1:.3f} ({recall1 * 100:.1f}%)")
        print(f"Recall@3:  {recall3:.3f} ({recall3 * 100:.1f}%)")
        print(f"Recall@5:  {recall5:.3f} ({recall5 * 100:.1f}%)")
        print(f"Recall@10: {recall10:.3f} ({recall10 * 100:.1f}%)")
