from pathlib import Path
import uuid
from src.chunking.chunks_generator import Chunking
from src.indexing_module.lexical_index import LexicalIndexing
from src.indexing_module.vector_index import VectorIndexing
from src.retrieval.retrieval import Retrieval
from src.answer_generation_modules.answer_generator import AnswerGenerator
from src.data_models.search_result import MinimalSearchResults
from src.error_handling_modules.inavlid_json import InvalidJSON
from src.evaluation_module.evaluation import Evaluation


class Pipeline:
    @staticmethod
    def run_pipeline() -> None:
        # 1 - chunking: write to data/processed/
        chunking = Chunking()
        chunks = chunking.apply_chunking()
        # 2 - indexing: save bm25 index to data/processed/bm25_index.pkl
        lexical_indexing = LexicalIndexing(chunks)
        bm25 = lexical_indexing.create_index()
        # Bonus: create chromadb index
        vector_indexing = VectorIndexing(chunks)
        collection = vector_indexing.create_index()
        # 3 - retrieval - retrieve relevant documents for all questions in
        # datasets_public/public/
        retrieval = Retrieval(chunks)
        
        retrieval.write_search_results()
        # Bonus: semantic retrieval
        answer_generator = AnswerGenerator()
        answer_generator.write_answers()
        evaluation = Evaluation()
        evaluation.print_report()
        # Bonus: serve app
        Pipeline.serve()

    @staticmethod
    def index(max_chunk_size: int = 2000,
              corpus_path: str = "data/raw",
              output_dir: str = "data/processed") -> None:
        try:
            chunking = Chunking(corpus_path, output_dir, max_chunk_size)
            chunks = chunking.apply_chunking()
            # 2 - indexing: save bm25 index to data/processed/bm25_index.pkl
            lexical_indexing = LexicalIndexing(chunks, output_dir)
            lexical_indexing.create_index()
            vector_indexing = VectorIndexing(chunks, output_dir)
            vector_indexing.create_index()
            print(f"Ingestion complete! Indexed {len(chunks)} chunks under {output_dir}/")
        except FileNotFoundError as error:
            print(error)

    @staticmethod
    def search(query: str,
               k: int = 5,
               index_dir: str = "data/processed") -> None:
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
    def search_dataset(dataset_path: str = "data/datasets/UnansweredQuestions/",
                       k: int = 5,
                       save_directory: str = "data/output/search_results",
                       index_dir: str = "data/processed") -> None:
        try:
            retrieval = Retrieval.from_index_dir(
                index_dir, k, dataset_path=dataset_path, save_directory=save_directory)
            retrieval.write_search_results()
            print(f"Saved student_search_results to {save_directory}/{Path(dataset_path).name}")
        except (FileNotFoundError, InvalidJSON) as error:
            print(error)

    @staticmethod
    def answer(query: str, k: int = 5,
               index_dir: str = "data/processed") -> None:
        if not query.strip():
            print("query must not be empty")
            return
        try:
            retrieval = Retrieval.from_index_dir(index_dir, k)
        except FileNotFoundError as error:
            print(error)
            return
        sources = retrieval.retrieve_context(query)
        generator = AnswerGenerator()
        question = MinimalSearchResults(question_id=str(uuid.uuid4()),
                                        question=query,
                                        retrieved_sources=sources)
        print(generator.answer_prompt(question))

    @staticmethod
    def answer_dataset(student_search_results_path: str,
                       save_directory: str = ".") -> None:
        try:
            generator = AnswerGenerator(student_search_results_path, save_directory)
            generator.write_answers()
            print("Saved student_search_results_and_answer to "
                  f"{save_directory}/{Path(student_search_results_path).name}")
        except (FileNotFoundError, InvalidJSON) as error:
            print(error)

    @staticmethod
    def evaluate(student_search_results_path: str, dataset_path: str) -> None:
        try:
            Evaluation(student_search_results_path, dataset_path).print_report()
        except (FileNotFoundError, InvalidJSON) as error:
            print(error)

    @staticmethod
    def serve(host: str = "0.0.0.0", port: int = 8000,
              index_dir: str = "data/processed") -> None:
        import uvicorn
        from src.api.app import create_app
        uvicorn.run(create_app(index_dir), host=host, port=port)
