"""Local HTTP API (bonus): exposes retrieval and answer generation over
plain HTTP, so the system can be driven by something other than the CLI.
Reuses the same `Retrieval`/`AnswerGenerator`
classes as the CLI — no duplicated logic.
"""
import uuid
from typing import List
from fastapi import FastAPI, HTTPException
from src.answer_generation_modules.answer_generator import AnswerGenerator
from src.data_models.minimal_source import MinimalSource
from src.data_models.search_answer import MinimalAnswer
from src.data_models.search_result import MinimalSearchResults
from src.retrieval_modules.retrieval import Retrieval
from src.models.language_model import LLM

def create_app(index_dir: str = "data/processed") -> FastAPI:
    """Build the FastAPI app exposing `/search` and `/answer`.

    Args:
        index_dir: Directory containing the persisted index (see `index`).
        model_path: HuggingFace repo id of the answer-generation model.
            Loaded once at startup and reused across requests.

    Returns:
        A configured FastAPI application.
    """
    app = FastAPI(title="RAG against the machine")

    @app.post("/search", response_model=List[MinimalSource])
    def search(query: str, k: int = 5) -> List[MinimalSource]:
        """Return the top-k sources for `query`."""
        if not query.strip():
            raise HTTPException(status_code=400, detail="query must not be empty")
        try:
            retrieval = Retrieval.from_index_dir(index_dir, k)
        except FileNotFoundError as error:
            raise HTTPException(status_code=503, detail=str(error))
        return retrieval.retrieve_context(query)

    @app.post("/answer", response_model=MinimalAnswer)
    def answer(query: str, k: int = 5) -> MinimalAnswer:
        """Retrieve context for `query` and generate a grounded answer."""
        if not query.strip():
            raise HTTPException(status_code=400, detail="query must not be empty")
        try:
            retrieval = Retrieval.from_index_dir(index_dir, k)
        except FileNotFoundError as error:
            raise HTTPException(status_code=503, detail=str(error))
        sources = retrieval.retrieve_context(query)
        question = MinimalSearchResults(question_id=str(uuid.uuid4()),
                                        question=query,
                                        retrieved_sources=sources)
        return MinimalAnswer(**question.model_dump(),
                             answer=LLM().chat())

    return app
