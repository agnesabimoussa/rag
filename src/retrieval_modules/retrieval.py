from src.data_models.unanswered_question import UnansweredQuestion
from src.data_models.minimal_source import MinimalSource
from src.error_handling_modules.inavlid_json import InvalidJSON
from src.data_models.search_result import (StudentSearchResults,
                                           MinimalSearchResults)
from src.chunking_modules.chunk import Chunk
from src.text_processing import tokenize_text
from rank_bm25 import BM25Okapi
from pathlib import Path
import heapq
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
        py_chunk_file = index_path / "code_chunks.json"
        if not py_chunk_file.is_file():
            py_chunk_file = index_path / "code_chunk_file.json"
        md_chunk_file = index_path / "markdown_chunks.json"
        if not md_chunk_file.is_file():
            md_chunk_file = index_path / "markdown_chunk_file.json"
        bm25_file = index_path / "bm25_index.pkl"
        if not py_chunk_file.is_file() or not bm25_file.is_file() or not md_chunk_file.is_file():
            raise FileNotFoundError(
                f"No index found under {index_dir}. Run the `index` command first."
            )
        # Order must match Chunking.apply_chunking(), which returns
        # markdown_chunks + code_chunks and is what the persisted BM25 index
        # (data/processed/bm25_index.pkl) was built from — get_scores()
        # returns positions in that fit order, so loading chunks back in a
        # different order here would silently pair scores with the wrong
        # chunk.
        chunks = []
        with md_chunk_file.open("r", encoding="utf-8") as file:
            chunks.extend(TypeAdapter(List[Chunk]).validate_python(json.load(file)))
        with py_chunk_file.open("r", encoding="utf-8") as file:
            chunks.extend(TypeAdapter(List[Chunk]).validate_python(json.load(file)))
        with bm25_file.open("rb") as file:
            bm25 = pickle.load(file)
        return cls(bm25, save_directory, dataset_path, chunks, k)

    # Find the best matching chunk, then use the remaining context slots to retrieve chunks
    # from the same original source so the LLM gets more surrounding context.
    def retrieve_context(self, prompt: str) -> List[MinimalSource]:
        tokenized_query = tokenize_text(prompt)
        if not tokenized_query:
            tokenized_query = prompt.lower().split()

        scores = self.bm25.get_scores(tokenized_query)
        candidate_pool = min(len(self.chunks), max(self.k * 8, 40))
        ranked_indices = [
            index
            for index, _ in heapq.nlargest(
                candidate_pool,
                enumerate(scores),
                key=lambda item: item[1],
            )
        ]
        ranked_chunks = [self.chunks[index] for index in ranked_indices]

        unique_chunks = []
        seen_groups = set()
        seen_chunk_ids = set()
        seen_source_spans = set()

        for chunk in ranked_chunks:
            group_id = chunk.original_chunk_id or chunk.id
            if group_id in seen_groups:
                continue
            source_span = (
                chunk.source,
                chunk.first_character_index,
                chunk.last_character_index,
            )
            if chunk.id not in seen_chunk_ids and source_span not in seen_source_spans:
                unique_chunks.append(chunk)
                seen_groups.add(group_id)
                seen_chunk_ids.add(chunk.id)
                seen_source_spans.add(source_span)
            if len(unique_chunks) == self.k:
                break

        if len(unique_chunks) < self.k:
            for chunk in ranked_chunks:
                source_span = (
                    chunk.source,
                    chunk.first_character_index,
                    chunk.last_character_index,
                )
                if chunk.id in seen_chunk_ids or source_span in seen_source_spans:
                    continue
                unique_chunks.append(chunk)
                seen_chunk_ids.add(chunk.id)
                seen_source_spans.add(source_span)
                if len(unique_chunks) == self.k:
                    break

        return [
            MinimalSource(
                file_path=chunk.source,
                first_character_index=chunk.first_character_index,
                last_character_index=chunk.last_character_index
            )
            for chunk in unique_chunks
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
