# import fire
from indexing.index_generator import Indexing


def main() -> None:
    indexing = Indexing("data/raw", "data/processed/", 256)
    indexing.write_result()
