# import fire
from src.pipeline_module.pipeline import Pipeline
# This will run the rag pipeline on the whole batch of data
# Put them in a class called pipeline instead


def main() -> None:
    # Source documents -> Chunking -> Indexing -> retrieval -> generation -> answer
    try:
        Pipeline.run_pipeline()
    except Exception as e:
        print(e)
