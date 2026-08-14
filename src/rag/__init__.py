# import fire
from chunking_modules.chunks_generator import Chunking
from indexing_module.chunks_indexing import Indexing

def main() -> None:
    # Source documents -> Chunking -> Indexing -> retrieval -> generation -> answer
    # try:
        # 1 - chunking: write to data/processed/index_file.json
        chunking = Chunking("data/raw", "data/processed/chunk_file.json")
        chunking.chunk_files()
        chunking.write_result()
        chunks_text = chunking.get_chunks_text()
        # 2 - indexing: save bm25 index to data/processed/bm25_index.pkl
        indexing = Indexing(chunks_text, "data/processed/bm25_index.pkl")
        indexing.create_index()
    # except Exception as e:
    #     print(e)
