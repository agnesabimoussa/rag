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
    """Lexical (BM25) retrieval over a pre-built chunk index.

    Can be used for a single ad-hoc query (`retrieve_context`) or over a
    whole question dataset (`search_dataset` / `write_search_results`),
    which require `dataset_path` and `save_directory` respectively.
    """

    def __init__(self,
                 bm25: BM25Okapi,
                 save_directory: Optional[str],
                 dataset_path: Optional[str],
                 chunks: List[Chunk],
                 k: int = 5) -> None:
        """Initialize retrieval over an already-built BM25 index.

        Args:
            bm25: BM25 index built over `chunks` (see `Indexing.create_index`).
            save_directory: Directory `write_search_results` writes into.
                Only required for dataset-batch usage.
            dataset_path: Path to a JSON file of `UnansweredQuestion`
                entries. Only required for dataset-batch usage.
            chunks: The chunks the BM25 index was built from, in the same
                order, so BM25 result indices map back to real sources.
            k: Number of sources to retrieve per query.
        """
        self.bm25 = bm25
        self.save_directory = Path(save_directory) if save_directory else None
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.chunks = chunks
        self.k = k
        self.max_chars = 2000

    @classmethod
    def from_index_dir(cls,
                       index_dir: str,
                       k: int = 5,
                       dataset_path: Optional[str] = None,
                       save_directory: Optional[str] = None) -> "Retrieval":
        """Reconstruct a `Retrieval` from an index persisted by `Indexing`.

        Each CLI command runs in its own process, so the BM25 index and its
        backing chunks must be reloaded from disk rather than passed
        in-memory from the `index` command.

        Args:
            index_dir: Directory containing `chunk_file.json` and
                `bm25_index.pkl`, as written by the `index` command.
            k: Number of sources to retrieve per query.
            dataset_path: Path to a JSON file of `UnansweredQuestion`
                entries, for dataset-batch usage.
            save_directory: Directory to write dataset-batch results into.

        Returns:
            A `Retrieval` ready to query.

        Raises:
            FileNotFoundError: If the index has not been built yet.
        """
        index_path = Path(index_dir)
        chunk_file = index_path / "chunk_file.json"
        bm25_file = index_path / "bm25_index.pkl"
        if not chunk_file.is_file() or not bm25_file.is_file():
            raise FileNotFoundError(
                f"No index found under {index_dir}. Run the `index` command first."
            )
        with chunk_file.open("r", encoding="utf-8") as file:
            chunks = TypeAdapter(List[Chunk]).validate_python(json.load(file))
        with bm25_file.open("rb") as file:
            bm25 = pickle.load(file)
        return cls(bm25, save_directory, dataset_path, chunks, k)

    def retrieve_context(self, prompt: str) -> List[MinimalSource]:
        """Return the top-k sources for a single query.

        Args:
            prompt: The natural-language query.

        Returns:
            Up to `self.k` `MinimalSource` results, most relevant first.
        """
        tokenized_query = prompt.lower().split()
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
        """Load the questions dataset pointed to by `self.dataset_path`.

        Raises:
            InvalidJSON: If the file is missing, not valid JSON, or does not
                match the expected schema.
        """
        assert self.dataset_path is not None, "dataset_path is required for dataset-batch usage"
        try:
            with open(self.dataset_path, "r", encoding="utf-8") as file:
                content = json.load(file)
            adapter = TypeAdapter(List[UnansweredQuestion])
            return adapter.validate_python(content["rag_questions"])
        except (ValidationError, json.JSONDecodeError, KeyError, TypeError):
            raise InvalidJSON("InvalidJSON exception occured. Make sure"
                              f"{self.dataset_path} contains valid JSON.")

    def search_dataset(self) -> StudentSearchResults:
        """Retrieve context for every question in `self.dataset_path`.

        Returns:
            A `StudentSearchResults` with one `MinimalSearchResults` per
            question.
        """
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
        """Run `search_dataset` and persist the results as JSON.

        Writes to `self.save_directory / self.dataset_path.name`.
        """
        assert self.save_directory is not None, "save_directory is required to write results"
        assert self.dataset_path is not None, "dataset_path is required to write results"
        search_results = self.search_dataset()
        path = self.save_directory
        path.mkdir(parents=True, exist_ok=True)
        full_path = path / self.dataset_path.name
        with open(full_path, "w", encoding="utf-8") as file:
            json.dump(search_results.model_dump(),
                      file,
                      indent=4)

    def get_k(self) -> int:
        """Return the configured number of sources retrieved per query."""
        return self.k
