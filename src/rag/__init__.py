# import fire
from indexing.index_generator import Indexing


def main() -> None:
    indexing = Indexing("data/raw", "data/processed/")
    indexing.write_result()
