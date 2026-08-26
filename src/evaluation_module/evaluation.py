from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from pydantic import TypeAdapter, ValidationError
from src.data_models.answered_question import AnsweredQuestion
from src.data_models.minimal_source import MinimalSource
from src.data_models.search_result import StudentSearchResults
from src.error_handling_modules.inavlid_json import InvalidJSON

DEFAULT_KS: Tuple[int, ...] = (1, 3, 5, 10)
DEFAULT_IOU_THRESHOLD = 0.05


def _iou(first: MinimalSource, second: MinimalSource) -> float:
    if first.file_path != second.file_path:
        return 0.0
    start = max(first.first_character_index, second.first_character_index)
    end = min(first.last_character_index, second.last_character_index)
    intersection = max(0, end - start)
    union_start = min(first.first_character_index, second.first_character_index)
    union_end = max(first.last_character_index, second.last_character_index)
    union = union_end - union_start
    return intersection / union if union > 0 else 0.0


def recall_at_k(retrieved: List[MinimalSource],
                correct: List[MinimalSource],
                k: int,
                iou_threshold: float = DEFAULT_IOU_THRESHOLD) -> float:
    if not correct:
        return 1.0
    top_k = retrieved[:k]
    found = sum(
        1 for gt in correct
        if any(_iou(gt, result) >= iou_threshold for result in top_k)
    )
    return found / len(correct)


class Evaluation:
    def __init__(self, student_search_results_path: str, dataset_path: str) -> None:
        self.student_search_results_path = Path(student_search_results_path)
        self.dataset_path = Path(dataset_path)
        self.student_results = self._load_student_results()
        self.ground_truth = self._load_ground_truth()

    def _load_student_results(self) -> StudentSearchResults:
        try:
            with open(self.student_search_results_path, "r", encoding="utf-8") as file:
                content = json.load(file)
            return TypeAdapter(StudentSearchResults).validate_python(content)
        except (ValidationError, json.JSONDecodeError, OSError):
            raise InvalidJSON("InvalidJSON exception occured."
                              f"{self.student_search_results_path} contains invalid JSON.")

    def _load_ground_truth(self) -> Dict[str, List[MinimalSource]]:
        try:
            with open(self.dataset_path, "r", encoding="utf-8") as file:
                content = json.load(file)
            questions = TypeAdapter(List[AnsweredQuestion]).validate_python(
                content["rag_questions"])
            return {question.question_id: question.sources for question in questions}
        except (ValidationError, json.JSONDecodeError, KeyError, TypeError, OSError):
            raise InvalidJSON("InvalidJSON exception occured."
                              f"{self.dataset_path} contains invalid JSON.")

    def report(self, ks: Optional[List[int]] = None) -> str:
        resolved_ks = list(ks) if ks else list(DEFAULT_KS)
        matched = [
            (result.retrieved_sources, self.ground_truth[result.question_id])
            for result in self.student_results.search_results
            if result.question_id in self.ground_truth
        ]
        lines = [f"Questions evaluated: {len(matched)}"]
        for k in resolved_ks:
            if not matched:
                lines.append(f"Recall@{k}: n/a (no matching question_ids)")
                continue
            mean_recall = sum(
                recall_at_k(retrieved, correct, k) for retrieved, correct in matched
            ) / len(matched)
            lines.append(f"Recall@{k}: {mean_recall:.3f} ({mean_recall * 100:.1f}%)")
        return "\n".join(lines)
