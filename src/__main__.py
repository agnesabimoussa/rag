"""Entry point for the RAG CLI package.

This module exposes the command-line interface defined by the pipeline
classes and is invoked via ``python -m src``.
"""

import fire

from src.pipeline.pipeline import Pipeline


if __name__ == "__main__":
    fire.Fire(Pipeline)
