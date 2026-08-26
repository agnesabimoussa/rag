import uuid
from pathlib import Path

from src.chunking_modules.chunks_generator import Chunking
from src.indexing_module.chunks_indexing import Indexing
from src.retrieval_modules.retrieval import Retrieval
from src.answer_generation_modules.answer_generator import AnswerGenerator, DEFAULT_MODEL
from src.data_models.search_result import MinimalSearchResults
from src.error_handling_modules.inavlid_json import InvalidJSON
from src.evaluation_module.evaluation import Evaluation


class Pipeline:
    """Fire-facing CLI: `uv run python -m src <command> [options]`.

    Exposes `index`, `search`, `search_dataset`, `answer`, `answer_dataset`,
    `evaluate` plus `serve` for the Local HTTP API bonus.
    Every command catches missing/malformed input and prints a message
    instead of letting an unhandled traceback crash the CLI.
    """

    @staticmethod
    def run_pipeline() -> None:
        """Run the full ingestion, retrieval, and answer-generation workflow.

        This legacy helper mirrors the batch pipeline used by the project
        entry points and persists the generated artifacts under the data
        directory.
        """
        # 1 - chunking: write to data/processed/
        chunking = Chunking("data/raw", "data/processed/")
        chunks = chunking.apply_chunking()
        # 2 - indexing: save bm25 index to data/processed/bm25_index.pkl
        indexing = Indexing(chunks, "data/processed/bm25_index.pkl")
        bm25 = indexing.create_index()
        # 3 - retrieval - retrieve relevant documents for all questions in
        # datasets_public/public/
        # save dir, questions file
        # retrieval = Retrieval(
        #     bm25,
        #     "data/output/search_results/UnansweredQuestions/",
        #     "data/datasets/UnansweredQuestions/dataset_code_public.json",
        #     chunks)
        # retrieval.write_search_results()
        # answer_generator = AnswerGenerator("data/output/search_results/UnansweredQuestions/dataset_code_public.json",
        #                                    "data/output/search_results_and_answer/UnansweredQuestions")
        # answer_generator.write_answers()

    @staticmethod
    def index(max_chunk_size: int = 2000,
              corpus_path: str = "data/raw",
              output_dir: str = "data/processed") -> None:
        """Ingest `corpus_path` and persist chunk + BM25 index files.

        Args:
            max_chunk_size: Maximum characters per chunk.
            corpus_path: Directory to recursively chunk (code + markdown).
            output_dir: Directory `chunk_file.json`/`bm25_index.pkl` are
                written to.
        """
        try:
            chunking = Chunking("data/raw", "data/processed/")
            chunks = chunking.apply_chunking()
            # 2 - indexing: save bm25 index to data/processed/bm25_index.pkl
            indexing = Indexing(chunks, "data/processed/bm25_index.pkl")
            bm25 = indexing.create_index()
            print(f"Ingestion complete! Indexed {len(chunks)} chunks under {output_dir}/")
        except FileNotFoundError as error:
            print(error)

    @staticmethod
    def search(query: str, k: int = 5, index_dir: str = "data/processed") -> None:
        """Return the top-k sources for a single query.

        Args:
            query: The natural-language question.
            k: Number of sources to return.
            index_dir: Directory containing the persisted index (see `index`).
        """
        if not query.strip():
            print("Query must not be empty.")
            return
        if k <= 0:
            print("k value must be strictly positive.")
            return 
        try:
            retrieval = Retrieval.from_index_dir(index_dir, k)
        except FileNotFoundError as error:
            print(error)
            return
        for source in retrieval.retrieve_context(query):
            print(f"{source.file_path} [{source.first_character_index}:{source.last_character_index}]")

    @staticmethod
    def search_dataset(dataset_path: str, k: int = 5,
                       save_directory: str = ".",
                       index_dir: str = "data/processed") -> None:
        """Run retrieval over a whole question dataset and persist results.

        Args:
            dataset_path: Path to an `UnansweredQuestion` dataset JSON file.
            k: Number of sources to return per question.
            save_directory: Directory the `StudentSearchResults` JSON is
                written to. Scope it by dataset (e.g. `.../UnansweredQuestions`)
                so runs over different datasets don't overwrite each other.
            index_dir: Directory containing the persisted index (see `index`).
        """
        try:
            retrieval = Retrieval.from_index_dir(
                index_dir, k, dataset_path=dataset_path, save_directory=save_directory)
            retrieval.write_search_results()
            print(f"Saved student_search_results to {save_directory}/{Path(dataset_path).name}")
        except (FileNotFoundError, InvalidJSON) as error:
            print(error)

    @staticmethod
    def answer(query: str, k: int = 5,
               index_dir: str = "data/processed",
               model_path: str = DEFAULT_MODEL) -> None:
        """Answer a single query using retrieved context.

        Args:
            query: The natural-language question.
            k: Number of sources to retrieve for context.
            index_dir: Directory containing the persisted index (see `index`).
            model_path: HuggingFace repo id of the model to use.
        """
        if not query.strip():
            print("query must not be empty")
            return
        try:
            retrieval = Retrieval.from_index_dir(index_dir, k)
        except FileNotFoundError as error:
            print(error)
            return
        sources = retrieval.retrieve_context(query)
        generator = AnswerGenerator(model_path=model_path)
        question = MinimalSearchResults(question_id=str(uuid.uuid4()),
                                        question=query,
                                        retrieved_sources=sources)
        print(generator.answer_prompt(question))

    @staticmethod
    def answer_dataset(student_search_results_path: str,
                       save_directory: str = ".",
                       model_path: str = DEFAULT_MODEL) -> None:
        """Generate answers for a dataset of search results.

        Args:
            student_search_results_path: Path to a `StudentSearchResults`
                JSON file (as written by `search_dataset`).
            save_directory: Directory the `StudentSearchResultsAndAnswer`
                JSON is written to.
            model_path: HuggingFace repo id of the model to use.
        """
        try:
            generator = AnswerGenerator(student_search_results_path, save_directory,
                                        model_path=model_path)
            generator.write_answers()
            print("Saved student_search_results_and_answer to "
                  f"{save_directory}/{Path(student_search_results_path).name}")
        except (FileNotFoundError, InvalidJSON) as error:
            print(error)

    @staticmethod
    def evaluate(student_search_results_path: str, dataset_path: str) -> None:
        """Report recall@k of student search results against ground truth.

        For your own iteration only: the official score used during the
        defense is computed by the moulinette, not this command.

        Args:
            student_search_results_path: Path to a `StudentSearchResults`
                JSON file (as written by `search_dataset`).
            dataset_path: Path to a ground-truth `AnsweredQuestions` JSON
                file.
        """
        try:
            print(Evaluation(student_search_results_path, dataset_path).report())
        except (FileNotFoundError, InvalidJSON) as error:
            print(error)

    @staticmethod
    def serve(host: str = "127.0.0.1", port: int = 8000,
              index_dir: str = "data/processed",
              model_path: str = DEFAULT_MODEL) -> None:
        """Start the Local HTTP API (bonus), exposing `/search` and `/answer`.

        Args:
            host: Address to bind to.
            port: Port to bind to.
            index_dir: Directory containing the persisted index (see `index`).
            model_path: HuggingFace repo id of the model to use.
        """
        import uvicorn
        from src.http_api_module.app import create_app
        uvicorn.run(create_app(index_dir, model_path), host=host, port=port)
