from data_models.unanswered_question import UnansweredQuestion
from data_models.answered_question import AnsweredQuestion
from data_models.minimal_source import MinimalSource
from error_handling_modules.inavlid_json import InvalidJSON
from data_models.rag_dataset import RagDataset
from data_models.search_result import StudentSearchResults
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
                 chunks: List[str],
                 k: int = 5) -> None:
        self.bm25 = bm25
        self.answered_questions_path = Path(answered_questions_path)
        self.unanswered_questions_path = Path(unanswered_questions_path)
        self.rag_dataset = Retrieval._read_dataset(
            self.answered_questions_path, self.unanswered_questions_path)
        self.chunks = chunks
        self.k = k
        self.max_chars = 2000

    @staticmethod
    def _load_questions_into_dataset(questions_folder: Path,
                                     adapter: TypeAdapter,
                                     dataset: RagDataset) -> None:
        for file_path in questions_folder.rglob("*"):
            if file_path.is_file() and file_path.suffix == ".json":
                with file_path.open("r", encoding="utf-8") as file:
                    content = json.load(file)
                questions = adapter.validate_python(content["rag_questions"])
                dataset.rag_questions.extend(questions)

    @staticmethod
    def _read_dataset(answered_questions_path: Path,
                      unanswered_questions_path: Path) -> RagDataset:
        dataset = RagDataset(rag_questions=[])
        unanswered_questions_adapter = TypeAdapter(list[UnansweredQuestion])
        answered_questions_adapter = TypeAdapter(list[AnsweredQuestion])
        if (not answered_questions_path.is_dir()
                or not unanswered_questions_path.is_dir()):
            raise FileNotFoundError(
                "Input prompts directory does not exist."
            )
        try:
            Retrieval._load_questions_into_dataset(
                answered_questions_path, answered_questions_adapter, dataset)
            Retrieval._load_questions_into_dataset(
                unanswered_questions_path, unanswered_questions_adapter, dataset)
            return dataset
        except ValidationError:
            raise InvalidJSON(
                "InvalidJSON exception occured. Make sure input prompts contain valid JSON.")

    # retrieve context for a single query
    def retrieve_context(self, prompt: str) -> MinimalSource:
        tokenized_query = prompt.lower().split()
        # params: query, documents, top k (nb of sources to retrieve at max)
        # the top_chunks are the answer strings
        top_chunks = self.bm25.get_top_n(
            tokenized_query, self.chunks, n=self.k)
        return MinimalSource()

    def _chunk_to_source(self, chunk_text: str) -> MinimalSource:
        # Should resolve a retrieved chunk back to the MinimalSource it came
        # from: look up which file the chunk was cut out of and the
        # (first_character_index, last_character_index) span it covers there.
        # Not possible yet: self.chunks only holds raw chunk text (List[str]),
        # the file/offset provenance is discarded upstream in Indexing, which
        # only receives Chunking.get_chunks_text(). That metadata needs to
        # flow through as Chunk objects (or a parallel lookup) instead.
        pass

    def _load_dataset(self, dataset_path: str) -> List[UnansweredQuestion]:
        # Should load ONE dataset JSON file (e.g.
        # data/datasets/UnansweredQuestions/dataset_docs_public.json) and
        # return its list of UnansweredQuestion, for search_dataset to run
        # retrieve_context over. Distinct from _read_dataset/
        # _load_questions_into_dataset above, which scan whole folders and
        # merge answered + unanswered questions together - not what
        # search_dataset's single --dataset_path argument needs.
        
        pass

    # retrieve context for all queries
    def search_dataset(self, dataset_path: str, k: int) -> StudentSearchResults:
        # Should: load the questions at dataset_path via _load_dataset,
        # call retrieve_context (or its list-returning counterpart) for
        # each question, wrap each result in a MinimalSearchResults
        # (question_id, question, retrieved_sources), and return them all
        # inside a StudentSearchResults(search_results=..., k=k). Writing
        # the JSON to --save_directory is the CLI layer's job, not this
        # method's.
        pass

