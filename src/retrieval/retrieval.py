from src.data_models.unanswered_question import UnansweredQuestion
from src.data_models.minimal_source import MinimalSource
from src.data_models.search_result import (StudentSearchResults,
                                           MinimalSearchResults)
from src.data_models.chunk import Chunk, CodeChunk, MarkdownChunk
from src.utils.text_processing import tokenize_text
from rank_bm25 import BM25Okapi
from pathlib import Path
import heapq
from typing import List, Optional
import pickle
from tqdm import tqdm
from src.utils.file_operations import FileOperations
from src.retrieval.lexical_retriever import LexicalRetriever


class Retrieval:
    def __init__(self,
                 bm25: BM25Okapi,
                 chunks: List[Chunk],
                 k: int = 5,
                 dataset_path: str = "data/datasets/UnansweredQuestions/",
                 save_directory: str = "data/output/search_results/") -> None:
        self.bm25 = bm25
        self.save_directory = Path(save_directory)
        self.dataset_path = Path(dataset_path)
        self.lexical_retriver =LexicalRetriever()
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
        chunks: List[Chunk] = []
        with md_chunk_file.open("r", encoding="utf-8") as file:
            chunks.extend(FileOperations.load_content(file, List[MarkdownChunk]))
        with py_chunk_file.open("r", encoding="utf-8") as file:
            chunks.extend(FileOperations.load_content(file, List[CodeChunk]))
        with bm25_file.open("rb") as file:
            bm25 = pickle.load(file)
        if not save_directory:
            save_directory = "data/output/search_results/"
        if not dataset_path:
            dataset_path = "data/datasets/UnansweredQuestions/"
        return cls(bm25, chunks, k, dataset_path, save_directory)

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
                last_character_index=chunk.last_character_index,
                scope=getattr(chunk, "type", None)
            )
            for chunk in unique_chunks
        ]

    def search_dataset(self, file: Optional[Path] = None) -> StudentSearchResults:
        file = file or self.dataset_path
        search_results = StudentSearchResults(search_results=[], k=self.k)
        unanswered_questions = FileOperations.load_content(file, List[UnansweredQuestion])
        for question in tqdm(unanswered_questions, desc=f"Searching {file.name}"):
            sources = self.retrieve_context(question.question)
            search_result = MinimalSearchResults(question=question.question,
                                                 question_id=question.question_id,
                                                 retrieved_sources=sources)
            search_results.search_results.append(search_result)
        return search_results

    def write_search_results(self) -> None:
        files = FileOperations.resolve_files(self.dataset_path, ".json")
        self.save_directory.mkdir(parents=True, exist_ok=True)
        for file in files:
            search_results = self.search_dataset(file)
            full_path = self.save_directory / file.name
            FileOperations.write_json(full_path, search_results)
