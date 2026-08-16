from src.data_models.search_result import (StudentSearchResults,
                                           MinimalSearchResults)
from src.data_models.search_answer import (StudentSearchResultsAndAnswer,
                                           MinimalAnswer)
from src.error_handling_modules.inavlid_json import InvalidJSON
from pathlib import Path
from typing import Optional
from pydantic import TypeAdapter, ValidationError
import json
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


class AnswerGenerator:
    """Generates grounded natural-language answers from retrieved sources
    using a local causal LM (default: Qwen/Qwen3-0.6B).

    Can be used for a single ad-hoc answer (`answer_prompt`) or over a whole
    `StudentSearchResults` dataset (`answer_dataset` / `write_answers`),
    which require `student_search_results_path`.
    """

    def __init__(self,
                 student_search_results_path: Optional[str] = None,
                 save_dir: Optional[str] = None,
                 system_prompt: Optional[str] = None,
                 model_path: str = DEFAULT_MODEL) -> None:
        """Load the model and, optionally, a batch of search results.

        Args:
            student_search_results_path: Path to a `StudentSearchResults`
                JSON file. Only required for dataset-batch usage.
            save_dir: Directory `write_answers` writes into. Only required
                for dataset-batch usage.
            system_prompt: System prompt prefix. Defaults to a generic one.
            model_path: HuggingFace repo id of the model to use. Weights are
                downloaded once into `models/<slug>/` on first use.
        """
        self.student_search_results_path = (
            Path(student_search_results_path) if student_search_results_path else None
        )
        self.save_dir = Path(save_dir) if save_dir else None
        if self.student_search_results_path:
            self.search_results: Optional[StudentSearchResults] = self._read_search_results()
            self.k: Optional[int] = self.search_results.k
        else:
            self.search_results = None
            self.k = None
        if not system_prompt:
            system_prompt = """A default prompt"""
        self.system_prompt = system_prompt
        local_weights_dir = self._ensure_local_weights(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(local_weights_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            local_weights_dir,
            torch_dtype="auto",
            device_map="auto",
        )

    @staticmethod
    def _ensure_local_weights(model_path: str) -> str:
        """Download `model_path` into `models/<slug>/` if not already present.

        Keeping weights in a project-local, gitignored directory (rather
        than the global HF cache) means a fresh checkout works with zero
        manual setup, matching the subject's expectation that the evaluator
        regenerates model weights itself.

        Args:
            model_path: HuggingFace repo id, e.g. "Qwen/Qwen3-0.6B".

        Returns:
            Local directory containing the downloaded weights.
        """
        slug = model_path.split("/")[-1].lower()
        local_dir = Path("models") / slug
        if not (local_dir / "config.json").is_file():
            snapshot_download(repo_id=model_path, local_dir=str(local_dir))
        return str(local_dir)

    def _read_search_results(self) -> StudentSearchResults:
        """Load and validate `self.student_search_results_path`.

        Raises:
            InvalidJSON: If the file is missing, not valid JSON, or does not
                match the `StudentSearchResults` schema.
        """
        assert self.student_search_results_path is not None
        try:
            adapter = TypeAdapter(StudentSearchResults)
            with open(self.student_search_results_path, "r", encoding="utf-8") as file:
                content = json.load(file)
            return adapter.validate_python(content)
        except (ValidationError, json.JSONDecodeError):
            raise InvalidJSON("InvalidJSON exception occured."
                              f"{self.student_search_results_path} contains invalid JSON.")

    def answer_prompt(self, question: MinimalSearchResults) -> str:
        """Generate a grounded answer for a single question.

        Args:
            question: The question and its retrieved sources.

        Returns:
            The model's natural-language answer.
        """
        retrieved_sources = ", ".join(
            f"{source.file_path} [{source.first_character_index}:{source.last_character_index}]"
            for source in question.retrieved_sources
        )
        system_prompt = self.system_prompt + ". Retrieved sources: " + retrieved_sources
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question.question}
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
            )
        answer: str = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )
        return str(answer)

    def answer_dataset(self) -> StudentSearchResultsAndAnswer:
        """Generate answers for every question in `self.search_results`.

        Returns:
            A `StudentSearchResultsAndAnswer` with one `MinimalAnswer` per
            question.
        """
        assert self.search_results is not None, "student_search_results_path is required"
        assert self.k is not None
        search_results_and_answer = StudentSearchResultsAndAnswer(k=self.k, search_results=[])
        for search_result in tqdm(self.search_results.search_results, desc="Generating answers"):
            answer = MinimalAnswer(**search_result.model_dump(),
                                   answer=self.answer_prompt(search_result))
            search_results_and_answer.search_results.append(answer)
        return search_results_and_answer

    def write_answers(self) -> None:
        """Run `answer_dataset` and persist the results as JSON.

        Writes to `self.save_dir / self.student_search_results_path.name`.
        """
        assert self.save_dir is not None, "save_dir is required to write results"
        assert self.student_search_results_path is not None, \
            "student_search_results_path is required to write results"
        answers = self.answer_dataset()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        full_path = self.save_dir / self.student_search_results_path.name
        with open(full_path, "w", encoding="utf-8") as file:
            json.dump(answers.model_dump(),
                      file,
                      indent=4)
