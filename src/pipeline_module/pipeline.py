import uuid
from pathlib import Path
from src.chunking_modules.chunks_generator import Chunking
from src.indexing_module.chunks_indexing import Indexing
from src.retrieval_modules.retrieval import Retrieval
from src.answer_generation_modules.answer_generator import AnswerGenerator
from src.data_models.search_result import MinimalSearchResults
from src.error_handling_modules.inavlid_json import InvalidJSON
from src.evaluation_module.evaluation import Evaluation


class Pipeline:
    @staticmethod
    def run_pipeline() -> None:
        # 1 - chunking: write to data/processed/
        chunking = Chunking("data/raw", "data/processed/", 2000)
        chunks = chunking.apply_chunking()
        # 2 - indexing: save bm25 index to data/processed/bm25_index.pkl
        indexing = Indexing(chunks, "data/processed/bm25_index.pkl")
        bm25 = indexing.create_index()
        # 3 - retrieval - retrieve relevant documents for all questions in
        # datasets_public/public/
        # save dir, questions file
        retrieval = Retrieval(
            bm25,
            "data/output/search_results/",
            "data/datasets/UnansweredQuestions/dataset_code_public.json",
            chunks,
            k=10)
        retrieval.write_search_results()
        retrieval = Retrieval(
            bm25,
            "data/output/search_results/",
            "data/datasets/UnansweredQuestions/dataset_docs_public.json",
            chunks,
            k=10)
        retrieval.write_search_results()
        # answer_generator = AnswerGenerator("data/output/search_results/dataset_code_public.json",
        #                                    "data/output/search_results_and_answer/")
        # answer_generator.write_answers()
        # answer_generator = AnswerGenerator("data/output/search_results/dataset_docs_public.json",
        #                                    "data/output/search_results_and_answer/")
        # answer_generator.write_answers()
        evaluation = Evaluation("data/output/search_results/dataset_docs_public.json",
                                "data/datasets/AnsweredQuestions/dataset_docs_public.json")
        evaluation.print_report()
        evaluation = Evaluation("data/output/search_results/dataset_code_public.json",
                                "data/datasets/AnsweredQuestions/dataset_code_public.json")
        evaluation.print_report()

    # @staticmethod
    # def index(max_chunk_size: int = 2000,
    #           corpus_path: str = "data/raw",
    #           output_dir: str = "data/processed") -> None:
    #     try:
    #         chunking = Chunking("data/raw", "data/processed/")
    #         chunks = chunking.apply_chunking()
    #         # 2 - indexing: save bm25 index to data/processed/bm25_index.pkl
    #         indexing = Indexing(chunks, "data/processed/bm25_index.pkl")
    #         bm25 = indexing.create_index()
    #         print(f"Ingestion complete! Indexed {len(chunks)} chunks under {output_dir}/")
    #     except FileNotFoundError as error:
    #         print(error)

    # @staticmethod
    # def search(query: str,
    #            k: int = 5,
    #            index_dir: str = "data/processed") -> None:
    #     if not query.strip():
    #         print("Query must not be empty.")
    #         return
    #     if k <= 0:
    #         print("k value must be strictly positive.")
    #         return
    #     try:
    #         retrieval = Retrieval.from_index_dir(index_dir, k)
    #     except FileNotFoundError as error:
    #         print(error)
    #         return
    #     for source in retrieval.retrieve_context(query):
    #         print(f"{source.file_path} [{source.first_character_index}:{source.last_character_index}]")

    # @staticmethod
    # def search_dataset(dataset_path: str,
    #                    k: int = 5,
    #                    save_directory: str = ".",
    #                    index_dir: str = "data/processed") -> None:
    #     try:
    #         retrieval = Retrieval.from_index_dir(
    #             index_dir, k, dataset_path=dataset_path, save_directory=save_directory)
    #         retrieval.write_search_results()
    #         print(f"Saved student_search_results to {save_directory}/{Path(dataset_path).name}")
    #     except (FileNotFoundError, InvalidJSON) as error:
    #         print(error)

    # @staticmethod
    # def answer(query: str, k: int = 5,
    #            index_dir: str = "data/processed") -> None:
    #     if not query.strip():
    #         print("query must not be empty")
    #         return
    #     try:
    #         retrieval = Retrieval.from_index_dir(index_dir, k)
    #     except FileNotFoundError as error:
    #         print(error)
    #         return
    #     sources = retrieval.retrieve_context(query)
    #     generator = AnswerGenerator()
    #     question = MinimalSearchResults(question_id=str(uuid.uuid4()),
    #                                     question=query,
    #                                     retrieved_sources=sources)
    #     print(generator.answer_prompt(question))

    # @staticmethod
    # def answer_dataset(student_search_results_path: str,
    #                    save_directory: str = ".",
    #                    model_path: str = DEFAULT_MODEL) -> None:
    #     try:
    #         generator = AnswerGenerator(student_search_results_path, save_directory,
    #                                     model_path=model_path)
    #         generator.write_answers()
    #         print("Saved student_search_results_and_answer to "
    #               f"{save_directory}/{Path(student_search_results_path).name}")
    #     except (FileNotFoundError, InvalidJSON) as error:
    #         print(error)

    # @staticmethod
    # def evaluate(student_search_results_path: str, dataset_path: str) -> None:
    #     try:
    #         print(Evaluation(student_search_results_path, dataset_path).report())
    #     except (FileNotFoundError, InvalidJSON) as error:
    #         print(error)

    # @staticmethod
    # def serve(host: str = "127.0.0.1", port: int = 8000,
    #           index_dir: str = "data/processed") -> None:
    #     import uvicorn
    #     from src.http_api_module.app import create_app
    #     uvicorn.run(create_app(index_dir, model="Qwen/Qwen3-0.6B"), host=host, port=port)
