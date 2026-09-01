"""RAG package bootstrap helpers.

This package exposes the top-level pipeline entry point used to run the
full ingestion, retrieval, and answer-generation workflow.
"""

from src.pipeline.pipeline import Pipeline


def main() -> None:
    """Run the full pipeline end-to-end.

    Returns:
        None. Any pipeline exception is printed and surfaced to the caller.
    """
    try:
        Pipeline.run_pipeline()
    except Exception as exc:  # pragma: no cover - CLI convenience wrapper
        print(exc)
