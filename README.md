*This project has been created as part of the 42 curriculum by aabi-mou*

# Description

**RAG against the machine** is a Retrieval-Augmented Generation system that answers
questions about a codebase (the [vLLM](https://github.com/vllm-project/vllm) repository).
It ingests the corpus into a searchable lexical index, retrieves the most relevant
snippets for a question, and generates a grounded natural-language answer from them
using a small local language model (`Qwen/Qwen3-0.6B`). Retrieval quality is measured
with recall@k.

Pipeline: `Source documents -> Chunking -> Indexing -> Retrieval -> Generation -> Answer`,
with an `Evaluate` step (recall@k) run alongside `Retrieval` for the student's own
iteration.

# System architecture

```
data/raw/  --Chunking-->  Chunk[]  --Indexing-->  BM25 index (data/processed/)
                                                        |
question --------------------------------------> Retrieval.retrieve_context()
                                                        |
                                              top-k MinimalSource[]
                                                        |
                                          AnswerGenerator.answer_prompt()
                                             (Qwen/Qwen3-0.6B, local)
                                                        |
                                                     answer
```

Each stage is a small, independently testable class:

- `Chunking` (`src/chunking_modules/`) — walks `data/raw/`, splits `.md`/`.py`
  files into `Chunk`s and persists them to `data/processed/chunk_file.json`.
- `Indexing` (`src/indexing_module/`) — tokenizes the chunks and builds/persists
  a BM25 index (`data/processed/bm25_index.pkl`).
- `Retrieval` (`src/retrieval_modules/`) — loads the persisted index
  (`Retrieval.from_index_dir`) and answers single queries or whole datasets.
- `AnswerGenerator` (`src/answer_generation_modules/`) — loads the local model
  (auto-downloaded to `models/` on first use) and generates grounded answers.
- `Evaluation` (`src/evaluation_module/`) — computes recall@k for the student's
  own iteration.
- `Pipeline` (`src/pipeline_module/`) — the CLI surface (Python Fire) tying the
  above together: `index`, `search`, `search_dataset`, `answer`, `answer_dataset`,
  `evaluate`, plus `serve` for the Local HTTP API bonus.
- `src/http_api_module/` — a FastAPI app (bonus) exposing `/search` and `/answer`
  over plain HTTP, reusing the exact same `Retrieval`/`AnswerGenerator` classes as
  the CLI.

All pydantic data models (`src/data_models/`) are shared, validated contracts
between every stage.

# Chunking strategy

Python files and Markdown pages don't break apart the same way, so two distinct
strategies are implemented (`src/chunking_modules/`):

- **Code chunking** (`code_chunking.py`): `langchain_text_splitters`'
  `RecursiveCharacterTextSplitter.from_language(Language.PYTHON, ...)`, which is
  aware of Python syntax and prefers to split on function/class boundaries rather
  than mid-statement.
- **Markdown chunking** (`markdown_chunking.py`): `MarkdownHeaderTextSplitter`
  first splits on `#`/`##`/`###` headers (keeping headers attached to their
  content), then a `RecursiveCharacterTextSplitter` further splits any section
  still over the size limit.

Both are capped at `--max_chunk_size` characters (default 2000, configurable via
the `index` command), and each resulting chunk's exact `(first_character_index,
last_character_index)` span in the original file is recovered by `_SpanLocator`
(`chunks_generator.py`) — including chunks the splitter reformatted (dropped
whitespace, re-flowed lines), by matching on a whitespace-normalized copy of the
text and mapping back to real offsets.

# Retrieval method

Retrieval uses **BM25** (`rank_bm25.BM25Okapi`), a classic lexical
information-retrieval algorithm based on term-frequency scoring. At index time,
every chunk's text is lowercased and whitespace-tokenized; the same tokenization
is applied to the query at search time. `Retrieval.retrieve_context` calls
`bm25.get_top_n(tokenized_query, chunks, n=k)` to get the k highest-scoring chunks,
returned as `MinimalSource` (file path + character span).

Because each CLI invocation is a separate process, `Retrieval.from_index_dir`
reloads the persisted chunks and BM25 index from `data/processed/` rather than
keeping them in memory across commands.

# Performance analysis

Measured with `uv run python -m src evaluate` against the public
`AnsweredQuestions` datasets (100 docs questions, 99 code questions, k=10):

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Target (Recall@5) |
|---|---|---|---|---|---|
| docs | 50.0% | 64.0% | **69.0%** | 71.0% | 80% |
| code | 11.1% | 14.1% | **16.2%** | 18.2% | 50% |

Both are currently below the subject's thresholds. Pure BM25 with naive
lowercase/whitespace tokenization struggles most on code questions, where the
vocabulary gap between a natural-language question and identifier-heavy source
(`snake_case`, camelCase, punctuation) is largest. Closing this gap (better
tokenization, e.g. splitting identifiers; semantic embeddings; hybrid
lexical+semantic ranking) is the natural next step but was out of scope for this
pass, which focused on building a correct, spec-compliant pipeline end to end.

Indexing the full corpus (~3,200 files, ~14,900 chunks) takes ~3 seconds, well
under the 5-minute budget. `search_dataset` over 100 questions takes ~2 seconds,
well under the 90-second/200-question budget.

# Design decisions

- **Local model inference over the HF Inference API.** `AnswerGenerator` runs
  `Qwen/Qwen3-0.6B` locally via `transformers`, auto-downloading weights into a
  project-local `models/` directory (gitignored) on first use. This avoids
  depending on a paid, rate-limited external API and keeps the evaluator's fresh
  checkout self-contained, at the cost of CPU-bound generation latency.
- **Pydantic everywhere data crosses a stage boundary.** Every JSON file the
  pipeline reads or writes is validated against an explicit pydantic model
  (`src/data_models/`), so malformed input fails fast with a clear `InvalidJSON`
  error instead of propagating silently.
- **Persisted index, not in-memory pipeline.** Since each CLI command is its own
  process, the BM25 index and chunks are always reloaded from
  `data/processed/` (`Retrieval.from_index_dir`) rather than assuming a single
  long-lived pipeline object — this is what makes `search`/`answer` usable as
  standalone, fast, single-query commands.
- **The Local HTTP API bonus reuses the CLI's classes directly** (`Retrieval`,
  `AnswerGenerator`) rather than duplicating logic, so both surfaces stay in sync
  by construction.

# Challenges faced

- The default model, `Qwen/Qwen3-0.6B`, isn't served by the Hugging Face
  Inference router's default provider routing, and once a working provider was
  found, the account's free inference credits were exhausted mid-development —
  resolved by switching to local `transformers` inference.
- A messy mix of two internal import conventions (`from data_models.X import Y`
  vs `from src.data_models.X import Y` across sibling modules — both happened to
  resolve at runtime via the project's editable install) made `mypy .` fail
  outright with "source file found twice under different module names". Fixed
  by standardizing every internal import on the `src.`-qualified form and adding
  `__init__.py` to every `src/` subpackage, which also fixes the ambiguity for
  good rather than just working around it.
- `flake8 .` and `mypy .` initially crashed entirely (not just warned) because,
  with no exclude configuration, they recursed into `.venv` and the ingested
  `data/raw/vllm-0.10.1` corpus itself. Added `.flake8` and `[tool.mypy]`
  (`pyproject.toml`) exclude rules scoping both to the project's own code.

# Instructions

```bash
# Install dependencies (uv is the required package manager)
make install          # = uv sync

# Run the whole legacy one-shot pipeline (chunk -> index -> retrieve -> answer)
make run              # = uv run python -m src

# Or drive each stage explicitly via the Fire CLI:
uv run python -m src index --max_chunk_size 2000
uv run python -m src search "<question>" --k 5
uv run python -m src search_dataset --dataset_path <path> --k 10 --save_directory <dir>
uv run python -m src answer "<question>" --k 5
uv run python -m src answer_dataset --student_search_results_path <path> --save_directory <dir>
uv run python -m src evaluate --student_search_results_path <path> --dataset_path <path>
uv run python -m src serve --port 8000     # Local HTTP API (bonus)

make debug             # run under pdb
make lint              # flake8 . && mypy . (subject-mandated flags)
make clean             # remove __pycache__ / .mypy_cache
```

Weights for `Qwen/Qwen3-0.6B` (~1.2 GB) are downloaded automatically into
`models/` the first time `answer`/`answer_dataset`/`serve` runs — no manual
setup step required.

# Example usage

```bash
$ uv run python -m src index --max_chunk_size 2000
Ingestion complete! Indexed 14874 chunks under data/processed/

$ uv run python -m src search "How to configure the OpenAI server?" --k 5
data/raw/vllm-0.10.1/docs/deployment/frameworks/dstack.md [1940:3168]
data/raw/vllm-0.10.1/examples/online_serving/openai_chat_completion_client_with_tools_required.py [0:565]
...

$ uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
Questions evaluated: 100
Recall@1: 0.500 (50.0%)
Recall@3: 0.640 (64.0%)
Recall@5: 0.690 (69.0%)
Recall@10: 0.710 (71.0%)

$ uv run python -m src serve --port 8000 &
$ curl 'http://127.0.0.1:8000/answer?query=How+to+load+a+lora+adapter&k=3'
{"question_id":"...","question":"How to load a lora adapter","retrieved_sources":[...],"answer":"..."}
```

# Resources

- [rank_bm25](https://github.com/dorianbrown/rank_bm25) — the BM25 implementation used for retrieval.
- [Okapi BM25 (Wikipedia)](https://en.wikipedia.org/wiki/Okapi_BM25) — background on the ranking algorithm.
- [langchain-text-splitters](https://python.langchain.com/docs/how_to/#text-splitters) — `RecursiveCharacterTextSplitter` / `MarkdownHeaderTextSplitter` used for chunking.
- [Pydantic docs](https://docs.pydantic.dev/) — data model validation.
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) / [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B) — local answer generation.
- [Python Fire](https://github.com/google/python-fire) — the CLI framework.
- [FastAPI](https://fastapi.tiangolo.com/) — the Local HTTP API bonus.

**How AI was used:** Claude Code (Anthropic) was used as a pair-programming
assistant throughout this project. It read the subject PDF and cross-checked the
implementation against it; helped diagnose and fix the `mypy`/`flake8` module
resolution and configuration issues described above; scaffolded the Fire CLI
(`Pipeline`), the recall@k evaluation module, and the Local HTTP API bonus
(FastAPI) from the subject's requirements; and added docstrings across the
codebase. All AI-suggested code was reviewed before being accepted, and changes
were made incrementally and verified by actually running the CLI commands
(`index`, `search`, `search_dataset`, `evaluate`, `answer`, `serve`) end to end
against the real vLLM corpus rather than assumed to work.
