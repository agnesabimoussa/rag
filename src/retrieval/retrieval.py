from src.data_models.unanswered_question import UnansweredQuestion
from src.data_models.search_result import (StudentSearchResults,
                                           MinimalSearchResults)
from src.data_models.chunk import Chunk, CodeChunk, MarkdownChunk
from rank_bm25 import BM25Okapi
from pathlib import Path
from typing import List, Optional
import pickle
from tqdm import tqdm
from src.utils.file_operations import FileOperations
from src.retrieval.lexical_retriever import LexicalRetriever
from src.retrieval.semantic_retriever import SemanticRetriever
from src.retrieval.hybrid_retrieval import HybridRetriever
from chromadb import Collection


class Retrieval:
    def __init__(self,
                 chunks: List[Chunk],
                 k: int = 5,
                 dataset_path: str = "data/datasets/UnansweredQuestions/",
                 save_directory: str = "data/output/search_results/") -> None:
        self.save_directory = Path(save_directory)
        self.dataset_path = Path(dataset_path)
        self.chunks = chunks
        self.k = k

    def get_lexical_rankings(self,
                             bm25: BM25Okapi):
        lexical_retriver = LexicalRetriever(bm25, self.chunks, self.k)

    def get_semantic_rankings(self, collection: Collection):
        semantic_retriever = SemanticRetriever(collection, self.k)

    def get_hybrid_rankings(self):
        hybrid_retriever = HybridRetriever()

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
