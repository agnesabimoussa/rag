from src.data_models.rag_dataset import RagDataset
from src.data_models.search_result import (StudentSearchResults,
                                           MinimalSearchResults)
from src.data_models.chunk import Chunk, CodeChunk, MarkdownChunk
from src.data_models.minimal_source import MinimalSource
from rank_bm25 import BM25Okapi
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle
import chromadb
from tqdm import tqdm
from src.utils.file_operations import FileOperations
from src.retrieval.lexical_retriever import LexicalRetriever
from src.retrieval.semantic_retriever import SemanticRetriever
from src.retrieval.hybrid_retrieval import HybridRetriever
from chromadb import Collection


class Retrieval:
    """Loads a persisted index and answers single queries or whole datasets.

    Combines the lexical (BM25) and semantic (Chroma) rankings via
    Reciprocal Rank Fusion (`HybridRetriever`). The semantic side degrades
    gracefully to lexical-only if no vector index was built.
    """

    def __init__(self,
                 chunks: List[Chunk],
                 bm25: BM25Okapi,
                 collection: Optional[Collection] = None,
                 k: int = 5,
                 dataset_path: str = "data/datasets/UnansweredQuestions/",
                 save_directory: str = "data/output/search_results/") -> None:
        self.save_directory = Path(save_directory)
        self.dataset_path = Path(dataset_path)
        self.chunks = chunks
        self.k = k
        self.lexical_retriever = LexicalRetriever(bm25, chunks, k)
        self.semantic_retriever = SemanticRetriever(collection, k) if collection is not None else None
        self.hybrid_retriever = HybridRetriever(chunks, k)
        # Bonus: cache query results. Cheap and most useful for a long-lived
        # process reusing one `Retrieval` across many calls (the `serve` API,
        # or repeated/duplicate questions in a dataset) rather than the CLI's
        # one-shot commands, which never revisit the same instance anyway.
        self._query_cache: Dict[Tuple[str, int], List[MinimalSource]] = {}

    def _pool_size(self, k: int) -> int:
        return min(len(self.chunks), max(k * 8, 40)) if self.chunks else 0

    def retrieve_context(self, prompt: str, k: Optional[int] = None) -> List[MinimalSource]:
        """Return the top-k sources for `prompt`, overriding the instance's
        default `k` for this call if given (e.g. per-request `k` in the
        HTTP API, while reusing one loaded index across requests)."""
        effective_k = k if k is not None else self.k
        cache_key = (prompt, effective_k)
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]

        pool_size = self._pool_size(effective_k)
        lexical_ids = self.lexical_retriever.rank_chunk_ids(prompt, pool_size)
        semantic_ids: List[str] = []
        if self.semantic_retriever is not None:
            semantic_ids = self.semantic_retriever.rank_chunk_ids(prompt, pool_size)
        sources = self.hybrid_retriever.combine(lexical_ids, semantic_ids, effective_k)
        self._query_cache[cache_key] = sources
        return sources

    @classmethod
    def from_index_dir(cls,
                       index_dir: str,
                       k: int = 5,
                       dataset_path: Optional[str] = None,
                       save_directory: Optional[str] = None) -> "Retrieval":
        index_path = Path(index_dir)
        py_chunk_file = index_path / "code_chunks.json"
        md_chunk_file = index_path / "markdown_chunks.json"
        bm25_file = index_path / "bm25_index.pkl"
        if not py_chunk_file.is_file() or not bm25_file.is_file() or not md_chunk_file.is_file():
            raise FileNotFoundError(
                f"No index found under {index_dir}. Run the `index` command first."
            )
        chunks: List[Chunk] = []
        chunks.extend(FileOperations.load_content(md_chunk_file, List[MarkdownChunk]))
        chunks.extend(FileOperations.load_content(py_chunk_file, List[CodeChunk]))
        with bm25_file.open("rb") as file:
            bm25 = pickle.load(file)

        collection = None
        chroma_path = index_path / "chroma_db"
        if chroma_path.is_dir():
            try:
                client = chromadb.PersistentClient(path=str(chroma_path))
                collection = client.get_collection(name="documents")
            except Exception:
                collection = None

        if not save_directory:
            save_directory = "data/output/search_results/"
        if not dataset_path:
            dataset_path = "data/datasets/UnansweredQuestions/"
        return cls(chunks, bm25, collection, k, dataset_path, save_directory)

    def search_dataset(self, file: Optional[Path] = None) -> StudentSearchResults:
        file = file or self.dataset_path
        search_results = StudentSearchResults(search_results=[], k=self.k)
        questions = FileOperations.load_content(file, RagDataset).rag_questions
        for question in tqdm(questions, desc=f"Searching {file.name}"):
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
