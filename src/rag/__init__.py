# import fire
from chunking_modules.chunks_generator import Chunking
from indexing_module.chunks_indexing import Indexing
from retrieval_modules.retrieval import Retrieval
from src.answer_generation_modules.answer_generator import AnswerGenerator
from src.pipeline_module.pipeline import Pipeline
# This will run the rag pipeline on the whole batch of data
# Put them in a class called pipeline instead


def main() -> None:
    # Source documents -> Chunking -> Indexing -> retrieval -> generation -> answer
    # try:
    # 1 - chunking: write to data/processed/index_file.json
    chunking = Chunking("data/raw", "data/processed/chunk_file.json")
    chunking.write_result()
    chunks = chunking.get_chunks()
    # 2 - indexing: save bm25 index to data/processed/bm25_index.pkl
    indexing = Indexing(chunks, "data/processed/bm25_index.pkl")
    bm25 = indexing.create_index()
    # 3 - retrieval - retrieve relevant documents for all questions in
    # datasets_public/public/
    # save dir, questions file
    retrieval = Retrieval(
        bm25,
        "data/output/search_results/UnansweredQuestions/",
        "data/datasets/UnansweredQuestions/dataset_code_public.json",
        chunks)
    retrieval.write_search_results()
    answer_generator = AnswerGenerator("data/output/search_results/UnansweredQuestions/dataset_code_public.json",
                                       "data/output/search_results_and_answer/UnansweredQuestions"
                                       )
    answers = answer_generator.answer_dataset()
    # except Exception as e:
    #     print(e)
