from src.data_models.unanswered_question import UnansweredQuestion
from src.data_models.minimal_source import MinimalSource
from src.error_handling_modules.inavlid_json import InvalidJSON
from src.data_models.search_result import (StudentSearchResults,
                                           MinimalSearchResults)
from src.chunking_modules.chunk import Chunk
from rank_bm25 import BM25Okapi
from pathlib import Path
from typing import List, Optional
import json
import pickle
from pydantic import TypeAdapter, ValidationError
from tqdm import tqdm


class Retrieval:
    def __init__(self,
                 bm25: BM25Okapi,
                 save_directory: str,
                 dataset_path: str,
                 chunks: List[Chunk],
                 k: int = 5) -> None:
        self.bm25 = bm25
        self.save_directory = Path(save_directory)
        self.dataset_path = Path(dataset_path)
        self.chunks = chunks
        if k <= 0:
            k = 5
        self.k = k

    @classmethod
    def from_index_dir(cls,
                       index_dir: str,
                       k: int = 5,
                       dataset_path: Optional[str] = None,
                       save_directory: Optional[str] = None) -> "Retrieval":
        index_path = Path(index_dir)
        py_chunk_file = index_path / "code_chunk_file.json"
        md_chunk_file = index_path / "markdown_chunk_file.json"
        bm25_file = index_path / "bm25_index.pkl"
        if not py_chunk_file.is_file() or not bm25_file.is_file() or not md_chunk_file:
            raise FileNotFoundError(
                f"No index found under {index_dir}. Run the `index` command first."
            )
        chunks = []
        with py_chunk_file.open("r", encoding="utf-8") as file:
            chunks.extend(TypeAdapter(List[Chunk]).validate_python(json.load(file)))
        with md_chunk_file.open("r", encoding="utf-8") as file:
            chunks.extend(TypeAdapter(List[Chunk]).validate_python(json.load(file)))
        with bm25_file.open("rb") as file:
            bm25 = pickle.load(file)
        return cls(bm25, save_directory, dataset_path, chunks, k)

    # Find the best matching chunk, then use the remaining context slots to retrieve chunks 
    # from the same original source so the LLM gets more surrounding context.
    def retrieve_context(self, prompt: str) -> List[MinimalSource]:
        tokenized_query = prompt.lower().split()
        top_chunk = self.bm25.get_top_n(
            tokenized_query,
            self.chunks,
            n=1
        )[0]
        top_chunks = [top_chunk]
        k = self.k - 1
        original_chunk_id = top_chunk.original_chunk_id

        if original_chunk_id:
            related_chunks = [
                chunk
                for chunk in self.chunks
                if chunk.original_chunk_id == original_chunk_id
                and chunk.id != top_chunk.id
            ]
            original_chunk = next(
                (
                    chunk
                    for chunk in self.chunks
                    if chunk.id == original_chunk_id
                ),
                None
            )
            if original_chunk and original_chunk.id != top_chunk.id:
                related_chunks.append(original_chunk)
            top_chunks.extend(related_chunks[:k])

        else:
            top_chunks.extend(
                self.bm25.get_top_n(
                    tokenized_query,
                    self.chunks,
                    n=k
                )
            )

        return [
            MinimalSource(
                file_path=chunk.source,
                first_character_index=chunk.first_character_index,
                last_character_index=chunk.last_character_index
            )
            for chunk in top_chunks
        ]

    def _load_dataset(self) -> List[UnansweredQuestion]:
        try:
            with open(self.dataset_path, "r", encoding="utf-8") as file:
                content = json.load(file)
            adapter = TypeAdapter(List[UnansweredQuestion])
            return adapter.validate_python(content["rag_questions"])
        except (ValidationError, json.JSONDecodeError, KeyError, TypeError):
            raise InvalidJSON("InvalidJSON exception occured. Make sure"
                              f"{self.dataset_path} contains valid JSON.")

    def search_dataset(self) -> StudentSearchResults:
        search_results = StudentSearchResults(search_results=[], k=self.k)
        unanswered_questions = self._load_dataset()
        for question in tqdm(unanswered_questions, desc="Searching dataset"):
            sources = self.retrieve_context(question.question)
            search_result = MinimalSearchResults(question=question.question,
                                                 question_id=question.question_id,
                                                 retrieved_sources=sources)
            search_results.search_results.append(search_result)
        return search_results

    def write_search_results(self) -> None:
        search_results = self.search_dataset()
        path = self.save_directory
        path.mkdir(parents=True, exist_ok=True)
        full_path = path / self.dataset_path.name
        with open(full_path, "w", encoding="utf-8") as file:
            json.dump(search_results.model_dump(),
                      file,
                      indent=4)
