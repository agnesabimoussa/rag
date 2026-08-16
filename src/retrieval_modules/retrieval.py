from data_models.unanswered_question import UnansweredQuestion
from data_models.answered_question import AnsweredQuestion
from data_models.minimal_source import MinimalSource
from error_handling_modules.inavlid_json import InvalidJSON
from data_models.rag_dataset import RagDataset
from data_models.search_result import (StudentSearchResults,
                                       MinimalSearchResults)
from chunking_modules.chunk import Chunk
from rank_bm25 import BM25Okapi
from pathlib import Path
from typing import List
import json
from pydantic import TypeAdapter, ValidationError


class Retrieval:
    def __init__(self,
                 bm25: BM25Okapi,
                 answered_questions_path: str,
                 unanswered_questions_path: str,
                 chunks: List[Chunk],
                 k: int = 5) -> None:
        self.bm25 = bm25
        self.answered_questions_path = Path(answered_questions_path)
        self.unanswered_questions_path = Path(unanswered_questions_path)
        # self.rag_dataset = Retrieval._read_dataset(
        #     self.answered_questions_path, self.unanswered_questions_path)
        self.chunks = chunks
        self.k = k
        self.max_chars = 2000

    # @staticmethod
    # def _load_questions_into_dataset(questions_folder: Path,
    #                                  adapter: TypeAdapter,
    #                                  dataset: RagDataset) -> None:
    #     for file_path in questions_folder.rglob("*"):
    #         if file_path.is_file() and file_path.suffix == ".json":
    #             with file_path.open("r", encoding="utf-8") as file:
    #                 content = json.load(file)
    #             questions = adapter.validate_python(content["rag_questions"])
    #             dataset.rag_questions.extend(questions)

    # @staticmethod
    # def _read_dataset(answered_questions_path: Path,
    #                   unanswered_questions_path: Path) -> RagDataset:
    #     dataset = RagDataset(rag_questions=[])
    #     unanswered_questions_adapter = TypeAdapter(list[UnansweredQuestion])
    #     answered_questions_adapter = TypeAdapter(list[AnsweredQuestion])
    #     if (not answered_questions_path.is_dir()
    #             or not unanswered_questions_path.is_dir()):
    #         raise FileNotFoundError(
    #             "Input prompts directory does not exist."
    #         )
    #     try:
    #         Retrieval._load_questions_into_dataset(
    #             answered_questions_path, answered_questions_adapter, dataset)
    #         Retrieval._load_questions_into_dataset(
    #             unanswered_questions_path, unanswered_questions_adapter, dataset)
    #         return dataset
    #     except ValidationError:
    #         raise InvalidJSON(
    #             "InvalidJSON exception occured. Make sure input prompts contain valid JSON.")

    # retrieve context for a single query
    def retrieve_context(self, prompt: str) -> List[MinimalSource]:
        tokenized_query = prompt.lower().split()
        # params: query, documents, top k (nb of sources to retrieve at max)
        # the top_chunks are the matching Chunk objects, carrying source/offsets
        top_chunks = self.bm25.get_top_n(
            tokenized_query, self.chunks, n=self.k)
        return [
            MinimalSource(
                file_path=chunk.source,
                first_character_index=chunk.first_character_index,
                last_character_index=chunk.last_character_index)
            for chunk in top_chunks
        ]

    def _load_dataset(self) -> List[UnansweredQuestion]:
        with open(self.unanswered_questions_path, "r", encoding="utf-8") as file:
            content = json.load(file)
        try:
            adapter = TypeAdapter(List[UnansweredQuestion])
            unanswered_questions = adapter.validate_python(
                content["rag_questions"])
            return unanswered_questions
        except ValidationError:
            raise InvalidJSON("InvalidJSON exception occured. Make sure"
                              f"{self.unanswered_questions_path} contains valid JSON.")

    # retrieve context for all queries
    def search_dataset(self) -> StudentSearchResults:
        search_results = StudentSearchResults(search_results=[], k=self.k)
        unanswered_questions = self._load_dataset()
        for question in unanswered_questions:
            sources = self.retrieve_context(question.question)
            search_result = MinimalSearchResults(question=question.question,
                                                 question_id=question.question_id,
                                                 retrieved_sources=sources)
            search_results.search_results.append(search_result)
        return search_results

    def write_search_results(self) -> None:
        search_results = self.search_dataset()
        path = self.answered_questions_path
        path.mkdir(parents=True, exist_ok=True)
        full_path = path / "search_results.txt"
        with open(full_path, "w", encoding="utf-8") as file:
            json.dump(search_results.model_dump(),
                      file,
                      indent=4)
