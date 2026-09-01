"""Local HTTP API (bonus): exposes retrieval and answer generation over
plain HTTP, so the system can be driven by something other than the CLI.
Reuses the same `Retrieval`/`AnswerGenerator`
classes as the CLI — no duplicated logic.
"""
import uuid
from typing import Dict, List, Optional, Tuple
from fastapi import FastAPI, HTTPException
from src.answer_generation.answer_generator import AnswerGenerator
from src.data_models.minimal_source import MinimalSource
from src.data_models.search_answer import MinimalAnswer
from src.data_models.search_result import MinimalSearchResults
from src.retrieval.retrieval import Retrieval


def create_app(index_dir: str = "data/processed") -> FastAPI:
    app = FastAPI(title="RAG against the machine")
    # Bonus: loaded once at startup rather than per-request, so the server
    # reuses the already-loaded index/BM25/Chroma collection and LLM across
    # requests instead of re-reading them from disk on every call.
    answer_generator = AnswerGenerator()
    retrieval: Optional[Retrieval] = None
    try:
        retrieval = Retrieval.from_index_dir(index_dir)
    except FileNotFoundError:
        retrieval = None
    # Bonus: cache repeated (query, k) answers; `retrieval.retrieve_context`
    # caches sources the same way.
    answer_cache: Dict[Tuple[str, int], MinimalAnswer] = {}

    def _require_retrieval() -> Retrieval:
        if retrieval is None:
            raise HTTPException(status_code=503,
                                detail=f"No index found under {index_dir}. Run the `index` command first.")
        return retrieval

    @app.post("/search", response_model=List[MinimalSource])
    def search(query: str, k: int = 5) -> List[MinimalSource]:
        """Return the top-k sources for `query`."""
        if not query.strip():
            raise HTTPException(status_code=400, detail="query must not be empty")
        if k <= 0:
            raise HTTPException(status_code=400, detail="k must be strictly positive")
        return _require_retrieval().retrieve_context(query, k)

    @app.post("/answer", response_model=MinimalAnswer)
    def answer(query: str, k: int = 5) -> MinimalAnswer:
        """Retrieve context for `query` and generate a grounded answer."""
        if not query.strip():
            raise HTTPException(status_code=400, detail="query must not be empty")
        if k <= 0:
            raise HTTPException(status_code=400, detail="k must be strictly positive")
        cache_key = (query, k)
        if cache_key in answer_cache:
            return answer_cache[cache_key]
        sources = _require_retrieval().retrieve_context(query, k)
        question = MinimalSearchResults(question_id=str(uuid.uuid4()),
                                        question=query,
                                        retrieved_sources=sources)
        result = MinimalAnswer(**question.model_dump(),
                               answer=answer_generator.answer_prompt(question))
        answer_cache[cache_key] = result
        return result

    return app
