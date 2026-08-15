# import fire
from chunking_modules.chunks_generator import Chunking
from indexing_module.chunks_indexing import Indexing
from retrieval_modules.retrieval import Retrieval

# This will run the rag pipeline on the whole batch of data


def main() -> None:
    # Source documents -> Chunking -> Indexing -> retrieval -> generation -> answer
    # try:
    # 1 - chunking: write to data/processed/index_file.json
    chunking = Chunking("data/raw", "data/processed/chunk_file.json")
    chunking.write_result()
    chunks_text = chunking.get_chunks_text()
    # 2 - indexing: save bm25 index to data/processed/bm25_index.pkl
    indexing = Indexing(chunks_text, "data/processed/bm25_index.pkl")
    bm25 = indexing.create_index()
    # 3 - retrieval - retrieve relevant documents for all questions in
    # datasets_public/public/
    retrieval = Retrieval(
        bm25,
        "data/datasets/AnsweredQuestions/",
        "data/datasets/UnansweredQuestions/",
        chunks_text)
    
    # except Exception as e:
    #     print(e)
